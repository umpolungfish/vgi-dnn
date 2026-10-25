//! Transport the native kernel's membership positions into the existing
//! VGI/IMASM register and read the actual three-order algebra.
use imasm_core::imasm16_3::{Reg16_3, leq_i, leq_t, leq_c, join_t, meet_t, join_c, meet_c};

fn transport(m: u8) -> Reg16_3 {
    Reg16_3 { big_t: m & 8 != 0, big_f: m & 4 != 0,
              small_t: m & 2 != 0, small_f: m & 1 != 0 }
}
fn back(x: Reg16_3) -> u8 {
    u8::from(x.small_f) | (u8::from(x.small_t) << 1) |
    (u8::from(x.big_f) << 2) | (u8::from(x.big_t) << 3)
}
fn main() {
    for x in 0..16 {
        assert_eq!(back(transport(x)), x);
        for y in 0..16 {
            let a = transport(x);
            let b = transport(y);
            println!("transport {x} {y} {} {} {} {} {} {} {} {}",
                back(a.union(b)), back(join_t(a,b)), back(meet_t(a,b)),
                back(join_c(a,b)), back(meet_c(a,b)),
                leq_t(a,b), leq_c(a,b), leq_i(a,b));
        }
    }
}
