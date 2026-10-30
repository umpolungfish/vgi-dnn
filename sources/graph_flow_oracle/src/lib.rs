#![cfg_attr(not(feature = "host"), no_std)]
use imasm_core::{classic::Token, flow::gate_out, imasm16_3::Reg16_3};

// Token is repr(u8), contiguous from Vinit through Rotat in the bound source.
// The host vocabulary probe checks the complete discriminant/name correspondence.
pub fn token(code: u8) -> Option<Token> {
    if code > Token::Rotat as u8 { return None; }
    Some(unsafe { core::mem::transmute::<u8,Token>(code) })
}
pub fn carrier(bits: u8) -> Reg16_3 {
    Reg16_3 { big_t: bits&1!=0, big_f: bits&2!=0, small_t: bits&4!=0, small_f: bits&8!=0 }
}
pub fn bits(value: Reg16_3) -> u8 {
    value.big_t as u8 | ((value.big_f as u8)<<1) | ((value.small_t as u8)<<2) | ((value.small_f as u8)<<3)
}
#[no_mangle]
pub extern "C" fn ig_flow_gate(code: u32, input: u32, seed: u32, slot: u32, fan: u32) -> u32 {
    if code>u8::MAX as u32 || input>15 || seed>15 || fan==0 || fan>3 || slot>=fan { return u32::MAX; }
    let Some(tok)=token(code as u8) else {return u32::MAX};
    if fan as usize>tok.arity().1 {return u32::MAX;}
    bits(gate_out(tok,carrier(input as u8),carrier(seed as u8),slot as usize,fan as usize)) as u32
}

#[cfg(not(feature="host"))]
mod bare {
    use core::alloc::{GlobalAlloc,Layout};
    struct NoAllocation;
    unsafe impl GlobalAlloc for NoAllocation {
        unsafe fn alloc(&self,_layout:Layout)->*mut u8 {core::ptr::null_mut()}
        unsafe fn dealloc(&self,_pointer:*mut u8,_layout:Layout) {}
    }
    #[global_allocator]
    static ALLOCATOR:NoAllocation=NoAllocation;
    #[panic_handler]
    fn panic(_info:&core::panic::PanicInfo)->! {loop{core::hint::spin_loop()}}
}
