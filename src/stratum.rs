use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::pow::{LBRY_HEADER_LEN, double_sha256, lbry_work_hash};

#[derive(Debug)]
pub struct Client {
    url: String,
    stream: TcpStream,
}

impl Client {
    pub fn connect(url: &str) -> Result<Self> {
        let endpoint = Endpoint::parse(url)?;
        let stream = TcpStream::connect((endpoint.host.as_str(), endpoint.port))
            .with_context(|| format!("failed to connect to {url}"))?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(1)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;

        Ok(Self {
            url: url.to_owned(),
            stream,
        })
    }

    pub fn smoke(mut self, wallet: &str, worker: &str, duration: Duration) -> Result<()> {
        println!("connected to {}", self.url);

        self.send(json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["lbc-miner/0.1.0"]
        }))?;

        self.send(json!({
            "id": 2,
            "method": "mining.authorize",
            "params": [wallet, worker]
        }))?;

        let started = Instant::now();
        let mut reader = BufReader::new(self.stream.try_clone()?);
        let mut line = String::new();

        while started.elapsed() < duration {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => bail!("pool closed the connection"),
                Ok(_) => self.print_message(line.trim())?,
                Err(err)
                    if err.kind() == ErrorKind::WouldBlock || err.kind() == ErrorKind::TimedOut => {
                }
                Err(err) => return Err(err).context("failed while reading Stratum message"),
            }
        }

        println!("smoke complete after {}s", duration.as_secs());
        Ok(())
    }

    pub fn work_smoke(self, wallet: &str, worker: &str, duration: Duration) -> Result<()> {
        let first = self.first_work(wallet, worker, duration, true)?;
        println!("nonce2: {}", hex::encode(&first.work.nonce2));
        println!("merkle: {}", hex::encode(first.work.merkle_root));
        println!("header: {}", hex::encode(first.work.header));
        println!(
            "zero_nonce_work_hash: {}",
            hex::encode(first.work.zero_nonce_hash())
        );
        if let Some(diff) = first.difficulty {
            println!("active_difficulty: {diff}");
        }
        Ok(())
    }

    pub fn first_work(
        mut self,
        wallet: &str,
        worker: &str,
        duration: Duration,
        verbose: bool,
    ) -> Result<FirstWork> {
        if verbose {
            println!("connected to {}", self.url);
        }

        self.send(json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["lbc-miner/0.1.0"]
        }))?;

        self.send(json!({
            "id": 2,
            "method": "mining.authorize",
            "params": [wallet, worker]
        }))?;

        let started = Instant::now();
        let mut reader = BufReader::new(self.stream.try_clone()?);
        let mut line = String::new();
        let mut session = None;
        let mut authorized = false;
        let mut difficulty = None;
        let mut pending_notify = None;

        while started.elapsed() < duration {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => bail!("pool closed the connection"),
                Ok(_) => {
                    let Some(event) = parse_message(line.trim())? else {
                        continue;
                    };

                    match event {
                        StratumEvent::Subscribed(info) => {
                            if verbose {
                                println!(
                                    "subscribed: extranonce1={} extranonce2_size={}",
                                    hex::encode(&info.extranonce1),
                                    info.extranonce2_size
                                );
                            }
                            session = Some(WorkSession::new(info));
                        }
                        StratumEvent::Authorized(ok) => {
                            if verbose {
                                println!("authorized: {ok}");
                            }
                            authorized = ok;
                        }
                        StratumEvent::Difficulty(diff) => {
                            if verbose {
                                println!("difficulty: {diff}");
                            }
                            difficulty = Some(diff);
                        }
                        StratumEvent::Notify(notify) => {
                            if verbose {
                                println!("job: id={} clean={}", notify.job_id, notify.clean);
                            }
                            pending_notify = Some(notify);
                        }
                        StratumEvent::Other(text) => {
                            if verbose {
                                println!("{text}");
                            }
                        }
                        StratumEvent::SubmitResult {
                            id,
                            accepted,
                            error,
                        } => {
                            if verbose {
                                println!(
                                    "submit result: id={id} accepted={accepted} error={}",
                                    error.unwrap_or_default()
                                );
                            }
                        }
                    }

                    if authorized {
                        if let (Some(session), Some(notify)) =
                            (session.as_mut(), pending_notify.as_ref())
                        {
                            let work = session.build_work(notify)?;
                            return Ok(FirstWork { work, difficulty });
                        }
                    }
                }
                Err(err)
                    if err.kind() == ErrorKind::WouldBlock || err.kind() == ErrorKind::TimedOut => {
                }
                Err(err) => return Err(err).context("failed while reading Stratum message"),
            }
        }

        bail!("timed out before a complete Stratum job was available")
    }

    pub fn mining_session(
        mut self,
        wallet: &str,
        worker: &str,
        timeout: Duration,
        verbose: bool,
    ) -> Result<MiningSession> {
        if verbose {
            println!("connected to {}", self.url);
        }

        self.send(json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["lbc-miner/0.1.0"]
        }))?;

        self.send(json!({
            "id": 2,
            "method": "mining.authorize",
            "params": [wallet, worker]
        }))?;

        let reader = BufReader::new(self.stream.try_clone()?);
        let mut session = MiningSession {
            wallet: wallet.to_owned(),
            writer: self.stream,
            reader,
            work_session: None,
            authorized: false,
            difficulty: 1.0,
            notify: None,
            current_work: None,
            next_id: 100,
        };

        let started = Instant::now();
        while started.elapsed() < timeout {
            let update = session.poll_updates(Duration::from_millis(500), verbose)?;
            if update.authorized && update.has_work {
                return Ok(session);
            }
        }

        bail!("timed out before a complete mining session was available")
    }

    fn send(&mut self, message: Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(&message)?;
        bytes.push(b'\n');
        self.stream.write_all(&bytes)?;
        Ok(())
    }

    fn print_message(&self, line: &str) -> Result<()> {
        if line.is_empty() {
            return Ok(());
        }

        match parse_message(line)? {
            Some(StratumEvent::Difficulty(diff)) => println!("difficulty: {diff}"),
            Some(StratumEvent::Notify(notify)) => {
                println!("job: id={} clean={}", notify.job_id, notify.clean)
            }
            Some(StratumEvent::Subscribed(info)) => println!(
                "subscribed: extranonce1={} extranonce2_size={}",
                hex::encode(info.extranonce1),
                info.extranonce2_size
            ),
            Some(StratumEvent::Authorized(ok)) => println!("authorized: {ok}"),
            Some(StratumEvent::SubmitResult {
                id,
                accepted,
                error,
            }) => println!(
                "submit result: id={id} accepted={accepted} error={}",
                error.unwrap_or_default()
            ),
            Some(StratumEvent::Other(text)) => println!("{text}"),
            None => {}
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct MiningSession {
    wallet: String,
    writer: TcpStream,
    reader: BufReader<TcpStream>,
    work_session: Option<WorkSession>,
    authorized: bool,
    difficulty: f64,
    notify: Option<Notify>,
    current_work: Option<StratumWork>,
    next_id: u64,
}

impl MiningSession {
    pub fn work(&self) -> Result<&StratumWork> {
        self.current_work
            .as_ref()
            .context("no current Stratum work is available")
    }

    pub fn roll_work(&mut self) -> Result<()> {
        self.rebuild_work()
    }

    pub fn difficulty(&self) -> f64 {
        self.difficulty
    }

    pub fn poll_updates(&mut self, wait: Duration, verbose: bool) -> Result<SessionUpdate> {
        let nonblocking = wait.is_zero();
        self.reader.get_ref().set_nonblocking(nonblocking)?;
        if !nonblocking {
            self.reader.get_ref().set_read_timeout(Some(wait))?;
        }

        let mut update = SessionUpdate {
            authorized: self.authorized,
            has_work: self.current_work.is_some(),
            ..SessionUpdate::default()
        };
        let mut line = String::new();

        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => bail!("pool closed the connection"),
                Ok(_) => {
                    let Some(event) = parse_message(line.trim())? else {
                        continue;
                    };
                    self.apply_event(event, &mut update, verbose)?;
                    if !nonblocking {
                        self.reader
                            .get_ref()
                            .set_read_timeout(Some(Duration::from_millis(1)))?;
                    }
                }
                Err(err)
                    if err.kind() == ErrorKind::WouldBlock || err.kind() == ErrorKind::TimedOut =>
                {
                    break;
                }
                Err(err) => return Err(err).context("failed while reading Stratum message"),
            }
        }

        update.authorized = self.authorized;
        update.has_work = self.current_work.is_some();
        Ok(update)
    }

    pub fn submit_share(&mut self, work: &StratumWork, nonce: u32) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;

        let request = json!({
            "params": [
                self.wallet.as_str(),
                work.job_id.as_str(),
                hex::encode(&work.nonce2),
                hex::encode(work.ntime),
                StratumWork::submit_nonce_hex(nonce),
            ],
            "id": id,
            "method": "mining.submit"
        });
        self.writer.set_nonblocking(false)?;
        send_json(&mut self.writer, request)?;
        Ok(id)
    }

    fn apply_event(
        &mut self,
        event: StratumEvent,
        update: &mut SessionUpdate,
        verbose: bool,
    ) -> Result<()> {
        match event {
            StratumEvent::Subscribed(info) => {
                if verbose {
                    println!(
                        "subscribed: extranonce1={} extranonce2_size={}",
                        hex::encode(&info.extranonce1),
                        info.extranonce2_size
                    );
                }
                self.work_session = Some(WorkSession::new(info));
            }
            StratumEvent::Authorized(ok) => {
                if verbose {
                    println!("authorized: {ok}");
                }
                self.authorized = ok;
            }
            StratumEvent::Difficulty(diff) => {
                if verbose {
                    println!("difficulty: {diff}");
                }
                self.difficulty = diff;
            }
            StratumEvent::Notify(notify) => {
                if verbose {
                    println!("job: id={} clean={}", notify.job_id, notify.clean);
                }
                update.new_block = self
                    .notify
                    .as_ref()
                    .is_some_and(|current| current.prev_hash != notify.prev_hash);
                update.job_id = Some(notify.job_id.clone());
                update.clean = notify.clean;
                self.notify = Some(notify);
                self.rebuild_work()?;
                update.new_work = true;
            }
            StratumEvent::SubmitResult {
                id,
                accepted,
                error,
            } => {
                if accepted {
                    update.accepted += 1;
                    update.accepted_ids.push(id);
                    if verbose {
                        println!("share accepted: id={id}");
                    }
                } else {
                    update.rejected += 1;
                    update
                        .rejected_shares
                        .push((id, error.clone().unwrap_or_default()));
                    if verbose {
                        println!(
                            "share rejected: id={id} error={}",
                            error.unwrap_or_default()
                        );
                    }
                }
            }
            StratumEvent::Other(text) => {
                if verbose {
                    println!("{text}");
                }
            }
        }

        Ok(())
    }

    fn rebuild_work(&mut self) -> Result<()> {
        if let (Some(work_session), Some(notify)) =
            (self.work_session.as_mut(), self.notify.as_ref())
        {
            self.current_work = Some(work_session.build_work(notify)?);
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct SessionUpdate {
    pub new_work: bool,
    pub new_block: bool,
    pub clean: bool,
    pub job_id: Option<String>,
    pub authorized: bool,
    pub has_work: bool,
    pub accepted: u64,
    pub rejected: u64,
    pub accepted_ids: Vec<u64>,
    pub rejected_shares: Vec<(u64, String)>,
}

#[derive(Debug, Clone)]
pub struct SubscribeInfo {
    pub extranonce1: Vec<u8>,
    pub extranonce2_size: usize,
}

#[derive(Debug, Clone)]
pub struct WorkSession {
    info: SubscribeInfo,
    next_nonce2: u64,
}

impl WorkSession {
    pub fn new(info: SubscribeInfo) -> Self {
        Self {
            info,
            next_nonce2: 0,
        }
    }

    pub fn build_work(&mut self, notify: &Notify) -> Result<StratumWork> {
        let nonce2 = self.next_nonce2_bytes();
        self.next_nonce2 = self.next_nonce2.wrapping_add(1);

        let mut coinbase = Vec::new();
        coinbase.extend_from_slice(&notify.coinbase1);
        coinbase.extend_from_slice(&self.info.extranonce1);
        coinbase.extend_from_slice(&nonce2);
        coinbase.extend_from_slice(&notify.coinbase2);

        let mut merkle_root = double_sha256(&coinbase);
        for branch in &notify.merkle_branches {
            let mut joined = Vec::with_capacity(64);
            joined.extend_from_slice(&merkle_root);
            joined.extend_from_slice(branch);
            merkle_root = double_sha256(&joined);
        }

        let merkle_for_header = flip32(&merkle_root);
        let mut header = Vec::with_capacity(LBRY_HEADER_LEN);
        header.extend_from_slice(&notify.version);
        header.extend_from_slice(&notify.prev_hash);
        header.extend_from_slice(&merkle_for_header);
        header.extend_from_slice(&notify.claim_trie);
        header.extend_from_slice(&notify.ntime);
        header.extend_from_slice(&notify.nbits);
        header.extend_from_slice(&[0u8; 4]);

        if header.len() != LBRY_HEADER_LEN {
            bail!(
                "assembled LBRY work header must be {LBRY_HEADER_LEN} bytes, got {}",
                header.len()
            );
        }

        let mut fixed_header = [0u8; LBRY_HEADER_LEN];
        fixed_header.copy_from_slice(&header);

        Ok(StratumWork {
            job_id: notify.job_id.clone(),
            nonce2,
            ntime: notify.ntime,
            merkle_root,
            header: fixed_header,
        })
    }

    fn next_nonce2_bytes(&self) -> Vec<u8> {
        let bytes = self.next_nonce2.to_le_bytes();
        bytes[..self.info.extranonce2_size].to_vec()
    }
}

#[derive(Debug, Clone)]
pub struct Notify {
    pub job_id: String,
    pub prev_hash: [u8; 32],
    pub claim_trie: [u8; 32],
    pub coinbase1: Vec<u8>,
    pub coinbase2: Vec<u8>,
    pub merkle_branches: Vec<[u8; 32]>,
    pub version: [u8; 4],
    pub nbits: [u8; 4],
    pub ntime: [u8; 4],
    pub clean: bool,
}

impl Notify {
    fn from_params(params: &[Value]) -> Result<Self> {
        let has_trie = params.len() == 10;
        if !has_trie {
            bail!("LBRY Stratum notify must include claim-trie hash");
        }

        let job_id = required_str(params, 0)?.to_owned();
        let prev_hash = hex_array::<32>(required_str(params, 1)?)?;
        let claim_trie = hex_array::<32>(required_str(params, 2)?)?;
        let coinbase1 = hex_vec(required_str(params, 3)?)?;
        let coinbase2 = hex_vec(required_str(params, 4)?)?;

        let branch_values = params
            .get(5)
            .and_then(Value::as_array)
            .context("mining.notify merkle branch array missing")?;
        let merkle_branches = branch_values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .context("merkle branch is not a string")
                    .and_then(hex_array::<32>)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            job_id,
            prev_hash,
            claim_trie,
            coinbase1,
            coinbase2,
            merkle_branches,
            version: hex_array::<4>(required_str(params, 6)?)?,
            nbits: hex_array::<4>(required_str(params, 7)?)?,
            ntime: hex_array::<4>(required_str(params, 8)?)?,
            clean: params.get(9).and_then(Value::as_bool).unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
pub struct StratumWork {
    pub job_id: String,
    pub nonce2: Vec<u8>,
    pub ntime: [u8; 4],
    pub merkle_root: [u8; 32],
    pub header: [u8; LBRY_HEADER_LEN],
}

#[derive(Debug, Clone)]
pub struct FirstWork {
    pub work: StratumWork,
    pub difficulty: Option<f64>,
}

impl StratumWork {
    pub fn zero_nonce_hash(&self) -> [u8; 32] {
        lbry_work_hash(&self.header)
    }

    pub fn with_nonce(&self, nonce: u32) -> [u8; LBRY_HEADER_LEN] {
        let mut header = self.header;
        header[108..112].copy_from_slice(&nonce.to_le_bytes());
        header
    }

    pub fn submit_nonce_hex(nonce: u32) -> String {
        hex::encode(nonce.to_le_bytes())
    }
}

#[derive(Debug, Clone)]
enum StratumEvent {
    Subscribed(SubscribeInfo),
    Authorized(bool),
    Difficulty(f64),
    Notify(Notify),
    SubmitResult {
        id: u64,
        accepted: bool,
        error: Option<String>,
    },
    Other(String),
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Endpoint {
    host: String,
    port: u16,
}

impl Endpoint {
    fn parse(url: &str) -> Result<Self> {
        let stripped = url
            .strip_prefix("stratum+tcp://")
            .or_else(|| url.strip_prefix("tcp://"))
            .unwrap_or(url)
            .trim_end_matches('/');

        let (host, port) = stripped
            .rsplit_once(':')
            .context("Stratum URL must include host:port")?;
        if host.is_empty() {
            bail!("Stratum host is empty");
        }

        Ok(Self {
            host: host.to_owned(),
            port: port.parse().context("Stratum port is invalid")?,
        })
    }
}

fn summarize_result(message: &Value) -> String {
    if let Some(error) = message.get("error") {
        if !error.is_null() {
            return format!("error={error}");
        }
    }

    match message.get("result") {
        Some(value) => value.to_string(),
        None => "<no result>".to_owned(),
    }
}

fn parse_message(line: &str) -> Result<Option<StratumEvent>> {
    if line.is_empty() {
        return Ok(None);
    }

    let message: Value = serde_json::from_str(line).context("pool sent invalid JSON")?;
    if let Some(method) = message.get("method").and_then(Value::as_str) {
        return match method {
            "mining.set_difficulty" => {
                let diff = message
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| params.first())
                    .and_then(Value::as_f64)
                    .context("mining.set_difficulty value missing")?;
                Ok(Some(StratumEvent::Difficulty(diff)))
            }
            "mining.notify" => {
                let params = message
                    .get("params")
                    .and_then(Value::as_array)
                    .context("mining.notify params missing")?;
                Ok(Some(StratumEvent::Notify(Notify::from_params(params)?)))
            }
            other => Ok(Some(StratumEvent::Other(format!("method: {other}")))),
        };
    }

    if message.get("id") == Some(&Value::from(1)) {
        let result = message
            .get("result")
            .and_then(Value::as_array)
            .context("mining.subscribe result missing")?;
        let extranonce1 = hex_vec(
            result
                .get(1)
                .and_then(Value::as_str)
                .context("mining.subscribe extranonce1 missing")?,
        )?;
        let extranonce2_size = result
            .get(2)
            .and_then(Value::as_u64)
            .context("mining.subscribe extranonce2 size missing")?
            as usize;
        return Ok(Some(StratumEvent::Subscribed(SubscribeInfo {
            extranonce1,
            extranonce2_size,
        })));
    }

    if message.get("id") == Some(&Value::from(2)) {
        return Ok(Some(StratumEvent::Authorized(
            message
                .get("result")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        )));
    }

    if let Some(id) = message.get("id").and_then(Value::as_u64) {
        if let Some(accepted) = message.get("result").and_then(Value::as_bool) {
            let error = message.get("error").and_then(|error| {
                if error.is_null() {
                    None
                } else {
                    Some(error.to_string())
                }
            });
            return Ok(Some(StratumEvent::SubmitResult {
                id,
                accepted,
                error,
            }));
        }
    }

    Ok(Some(StratumEvent::Other(format!(
        "message: {}",
        summarize_result(&message)
    ))))
}

fn send_json(stream: &mut TcpStream, message: Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(&message)?;
    bytes.push(b'\n');
    stream.write_all(&bytes)?;
    Ok(())
}

fn required_str(params: &[Value], index: usize) -> Result<&str> {
    params
        .get(index)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string param {index}"))
}

fn hex_vec(input: &str) -> Result<Vec<u8>> {
    if input.len() % 2 != 0 {
        bail!("hex string has odd length");
    }
    hex::decode(input).context("invalid hex string")
}

fn hex_array<const N: usize>(input: &str) -> Result<[u8; N]> {
    let bytes = hex_vec(input)?;
    if bytes.len() != N {
        bail!("hex string must decode to {N} bytes, got {}", bytes.len());
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn flip32(input: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (src, dst) in input.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stratum_tcp_url() {
        let endpoint = Endpoint::parse("stratum+tcp://lbrypool.net:3334").unwrap();
        assert_eq!(endpoint.host, "lbrypool.net");
        assert_eq!(endpoint.port, 3334);
    }

    #[test]
    fn parses_bare_host_port() {
        let endpoint = Endpoint::parse("lbrypool.net:3334").unwrap();
        assert_eq!(endpoint.host, "lbrypool.net");
        assert_eq!(endpoint.port, 3334);
    }

    #[test]
    fn parses_lbry_notify_and_builds_header() {
        let params = serde_json::json!([
            "job",
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "0102",
            "0304",
            ["3333333333333333333333333333333333333333333333333333333333333333"],
            "20000000",
            "1a0e988c",
            "6a20714e",
            true
        ]);
        let notify = Notify::from_params(params.as_array().unwrap()).unwrap();
        let mut session = WorkSession::new(SubscribeInfo {
            extranonce1: hex::decode("8100ece2").unwrap(),
            extranonce2_size: 4,
        });

        let work = session.build_work(&notify).unwrap();
        assert_eq!(work.job_id, "job");
        assert_eq!(work.nonce2, [0, 0, 0, 0]);
        assert_eq!(work.header.len(), LBRY_HEADER_LEN);
        assert_eq!(&work.header[0..4], &[0x20, 0, 0, 0]);
        assert_eq!(&work.header[68..100], &[0x22; 32]);
        assert_eq!(&work.header[100..104], &[0x6a, 0x20, 0x71, 0x4e]);
        assert_eq!(&work.header[104..108], &[0x1a, 0x0e, 0x98, 0x8c]);
        assert_eq!(&work.header[108..112], &[0, 0, 0, 0]);
        assert_eq!(StratumWork::submit_nonce_hex(0x7856_3412), "12345678");
    }
}
