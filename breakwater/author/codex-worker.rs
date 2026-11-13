//! Private, single-purpose native executor. There is no agent/model loop here.
//! Only the author can start it; drop privilege before creating any runtime or
//! accepting RPC input. The model cannot select a listener or another identity.
use codex_exec_server::ExecServerRuntimePaths;
use codex_http_client::HttpClientFactory;
use codex_http_client::OutboundProxyPolicy;
use std::io;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

fn trusted(path: &str) -> io::Result<Option<Vec<u8>>> {
    use std::io::Read;
    let mut file = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let info = file.metadata()?;
    if !info.is_file() || info.uid() != 0 || info.mode() & 0o022 != 0 || info.len() > 1048576 {
        return Err(io::Error::other("Untrusted worker configuration"));
    }
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    Ok(Some(data))
}

fn checked(result: libc::c_int) -> io::Result<()> {
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Only root-owned bootstrap configuration is read before dropping privileges.
    unsafe {
        if libc::getuid() != 1100 || libc::geteuid() != 0 {
            return Err("Executor requires the author-to-worker entrypoint".into());
        }
        let capture = trusted("/run/breakwater/external-capture-enabled")?.is_some();
        if capture {
            let library = libc::dlopen(c"/opt/breakwater-runtime/stdio_capture.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if library.is_null() {
                return Err("Protected stdio capture could not be loaded".into());
            }
            let symbol = libc::dlsym(library, c"breakwater_capture_start".as_ptr());
            if symbol.is_null() {
                return Err("Protected stdio capture entrypoint is missing".into());
            }
            let start: unsafe extern "C" fn() -> libc::c_int = std::mem::transmute(symbol);
            checked(start())?;
            libc::dlclose(library);
            checked(libc::raise(libc::SIGSTOP))?;
        }
        checked(libc::clearenv())?;
        let groups: Vec<libc::gid_t> = match trusted("/run/breakwater/worker-groups")? {
            Some(data) => std::str::from_utf8(&data)?
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()?,
            None => vec![1200],
        };
        if groups.is_empty() || groups.len() > 64 || groups.contains(&0) || groups.contains(&1100) {
            return Err("Invalid worker groups".into());
        }
        checked(libc::setgroups(groups.len(), groups.as_ptr()))?;
        checked(libc::setresgid(1101, 1101, 1101))?;
        checked(libc::setresuid(1101, 1101, 1101))?;
        checked(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))?;
        // The executor retains a trusted RPC connection to the author. Worker
        // commands must not read its memory/descriptors or replace RPC replies.
        checked(libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0))?;
        checked(libc::close_range(3, u32::MAX, 0))?;
        libc::umask(0o002);
        for (key, value) in [
            ("HOME", "/home/worker"),
            ("USER", "worker"),
            ("LOGNAME", "worker"),
            ("PATH", "/usr/local/bin:/usr/bin:/bin"),
            ("SHELL", "/bin/bash"),
            ("LANG", "C.UTF-8"),
        ] {
            std::env::set_var(key, value);
        }
        if let Some(data) = trusted("/run/breakwater/worker.env")? {
            if data.last() != Some(&0) {
                return Err("Invalid worker environment".into());
            }
            for entry in data.split(|b| *b == 0).filter(|entry| !entry.is_empty()) {
                let (key, value) = std::str::from_utf8(entry)?
                    .split_once('=')
                    .ok_or("Invalid worker variable")?;
                std::env::set_var(key, value);
            }
        }
    }
    if std::env::args_os().len() != 1 {
        return Err("Executor does not accept arguments".into());
    }
    std::env::set_current_dir(std::env::var("BREAKWATER_WORKDIR").unwrap_or_else(|_| "/app".into()))?;
    // Hidden helper re-entry is deliberately unavailable. The profile uses the
    // container/UID boundary and native unsandboxed worker IO, without escalation.
    let paths = ExecServerRuntimePaths::new(
        "/opt/author-cli/codex-worker".into(),
        /*codex_linux_sandbox_exe*/ None,
    )?;
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(codex_exec_server::run_main(
            "stdio",
            paths,
            HttpClientFactory::new(OutboundProxyPolicy::ReqwestDefault),
        ))
}
