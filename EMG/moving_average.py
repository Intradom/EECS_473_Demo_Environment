class moving_avg_filter_t:
    def __init__(self, n_fast_mv_avg_count, n_slow_mv_avg_count, triggerThreshold, n_trigger_hold):
        self.prev_values_fast = list()
        self.n_fast_mv_avg_count = n_fast_mv_avg_count
        self.index_fast = 0
        self.avg_value_fast = 0

        self.prev_values_slow = list()
        self.n_slow_mv_avg_count = n_slow_mv_avg_count
        self.index_slow = 0
        self.avg_value_slow = 0

        self.triggerThreshold = triggerThreshold
        self.n_trigger_hold = n_trigger_hold
        self.isTriggered = False
        self.triggeredSamples = 0

        self.diff_sample = None

    def fast_filter(self, sample_in):
        if len(self.prev_values_fast)==0:
            self.prev_values_fast = [1*sample_in]*self.n_fast_mv_avg_count
        
        prev_value = self.prev_values_fast[self.index_fast]
        self.prev_values_fast[self.index_fast] = sample_in
        self.index_fast = (self.index_fast+1)%self.n_fast_mv_avg_count
        self.avg_value_fast = self.avg_value_fast + (sample_in - prev_value)/self.n_fast_mv_avg_count
        return self.avg_value_fast

    def slow_filter(self, sample_in):
        if len(self.prev_values_slow)==0:
            self.prev_values_slow = [1*sample_in]*self.n_slow_mv_avg_count
        
        prev_value = self.prev_values_slow[self.index_slow]
        self.prev_values_slow[self.index_slow] = sample_in
        self.index_slow = (self.index_slow+1)%self.n_slow_mv_avg_count
        self.avg_value_slow = self.avg_value_slow + (sample_in - prev_value)/self.n_slow_mv_avg_count
        return self.avg_value_slow

    def signal_detect(self, sample_in):
        if self.diff_sample:
            sample = sample_in - self.diff_sample
        else:
            sample = 0
            self.diff_sample = sample_in
        fast_filter_op = self.fast_filter(sample)
        slow_filter_op = self.slow_filter(fast_filter_op)

        detection_value = 1
        if self.isTriggered:
            self.triggeredSamples += 1
            if self.triggeredSamples >= self.n_trigger_hold:
                self.isTriggered = False
                detection_value = 0
                self.triggeredSamples = 0
        else:
            if slow_filter_op <= self.triggerThreshold:
                self.isTriggered = True
            else:
                self.isTriggered = False
                detection_value = 0
        return [sample, fast_filter_op, slow_filter_op, detection_value]