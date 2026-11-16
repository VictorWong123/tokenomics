# CPU and host checks for Metal verification splitting

See `summary.json` for scope and counts. No Metal device was executed.
The context probe uses generated context code with synthetic CPU models and
checks full logits, rejected-suffix rollback and cache positions.

The four context logs are stored as JSON wrappers because upstream log lines
contain trailing spaces. Each wrapper preserves the original UTF-8 bytes in
`text` and records their SHA256 and length. Decode `text` and encode UTF-8 to
recover the exact original log. The original `.log` payloads also remain in
commit `2b55bd2`; the wrapper conversion does not discard or alter evidence.
