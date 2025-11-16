#ifndef TIMER_H
#define TIMER_H

/**
 * @brief Simple timer utility for performance measurement
 * 
 * Call timer_start() to begin timing, timer_stop() to get elapsed time.
 * 
 * @param stop 0 to start timer, 1 to stop and get elapsed time
 * @return 0.0 when starting, elapsed seconds when stopping
 */
double record_time(int stop);

#endif // TIMER_H
