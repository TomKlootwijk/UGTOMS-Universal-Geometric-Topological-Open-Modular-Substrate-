def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0][:]]  # Make a copy to avoid aliasing
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlapping or adjacent intervals
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current[:])  # Make a copy to avoid aliasing
    
    return merged
