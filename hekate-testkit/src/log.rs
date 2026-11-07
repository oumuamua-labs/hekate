// SPDX-FileCopyrightText: 2026 Andrei Kochergin <andrei@oumuamua.dev>
// SPDX-FileCopyrightText: 2026 Oumuamua Labs <info@oumuamua.dev>
// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::RefCell;
use std::io::Write;
use std::sync::Once;

thread_local! {
    static LOG: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

struct ThreadLog;

impl Write for ThreadLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        LOG.with(|log| log.borrow_mut().extend_from_slice(buf));

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// One global subscriber, each thread logging to its own buffer:
/// a scoped subscriber misses an event whose callsite another test
/// thread, running with no subscriber, cached as uninterested.
pub fn logged<T>(verify: impl FnOnce() -> T) -> (T, String) {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(|| ThreadLog)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("a global subscriber is already set");
    });

    LOG.with(|log| log.borrow_mut().clear());

    let result = verify();
    let text = LOG.with(|log| String::from_utf8_lossy(&log.borrow()).into_owned());

    (result, text)
}
