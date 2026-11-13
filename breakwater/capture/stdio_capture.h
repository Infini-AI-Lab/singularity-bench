/* Called only by a trusted, single-threaded root launcher, before uid drop.
 * Returns in the child. The parent remains a protected stream supervisor.
 */
int breakwater_capture_start(void);
