struct IntDivider {
    uint divisor;
    uint multiplier;
    uint shift_val;
    uint pad;
};

uint fast_div(uint n, IntDivider d) {
    uint t = (uint)(((uint64_t)n * d.multiplier) >> 32);
    return (t + n) >> d.shift_val;
}