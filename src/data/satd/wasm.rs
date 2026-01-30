use core::arch::wasm32::*;

use v_frame::pixel::Pixel;

use crate::data::plane::PlaneRegion;

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn get_satd_internal<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    if std::mem::size_of::<T>() != 1 || w != 8 || h != 8 {
        return super::rust::get_satd_internal(src, dst, w, h, bit_depth);
    }

    let stride1 = src.plane_cfg.stride.get();
    let stride2 = dst.plane_cfg.stride.get();

    // SAFETY: We verified size_of::<T>() == 1.
    let ptr1 = src[0].as_ptr() as *const u8;
    let ptr2 = dst[0].as_ptr() as *const u8;

    let mut v = [i32x4_splat(0); 8];
    for i in 0..8 {
        let r1 = v128_load64_zero(ptr1.add(i * stride1) as *const u64);
        let r2 = v128_load64_zero(ptr2.add(i * stride2) as *const u64);

        let r1_16 = u16x8_extend_low_u8x16(r1);
        let r2_16 = u16x8_extend_low_u8x16(r2);
        v[i] = i16x8_sub(r1_16 as v128, r2_16 as v128);
    }

    hadamard_butterfly(&mut v);
    transpose_8x8_i16(&mut v);
    hadamard_butterfly(&mut v);

    let mut sum = u32x4_splat(0);
    for i in 0..8 {
        let abs = i16x8_abs(v[i]);
        sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(abs));
    }

    (i32x4_extract_lane::<0>(sum)
        + i32x4_extract_lane::<1>(sum)
        + i32x4_extract_lane::<2>(sum)
        + i32x4_extract_lane::<3>(sum)) as u32
        / 2
}

unsafe fn hadamard_butterfly(v: &mut [v128; 8]) {
    for i in (0..8).step_by(2) {
        let a = v[i];
        let b = v[i + 1];
        v[i] = i16x8_add(a, b);
        v[i + 1] = i16x8_sub(a, b);
    }
    for i in (0..8).step_by(4) {
        for j in 0..2 {
            let a = v[i + j];
            let b = v[i + j + 2];
            v[i + j] = i16x8_add(a, b);
            v[i + j + 2] = i16x8_sub(a, b);
        }
    }
    for i in 0..4 {
        let a = v[i];
        let b = v[i + 4];
        v[i] = i16x8_add(a, b);
        v[i + 4] = i16x8_sub(a, b);
    }
}

unsafe fn transpose_8x8_i16(v: &mut [v128; 8]) {
    let t0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[0], v[1]);
    let t1 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[2], v[3]);
    let t2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[4], v[5]);
    let t3 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[6], v[7]);

    let t4 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[0], v[1]);
    let t5 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[2], v[3]);
    let t6 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[4], v[5]);
    let t7 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[6], v[7]);

    let m0 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t0, t1);
    let m1 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t0, t1);
    let m2 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t2, t3);
    let m3 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t2, t3);

    let m4 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t4, t5);
    let m5 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t4, t5);
    let m6 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t6, t7);
    let m7 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t6, t7);

    v[0] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m0, m2);
    v[1] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m0, m2);
    v[2] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m1, m3);
    v[3] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m1, m3);

    v[4] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m4, m6);
    v[5] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m4, m6);
    v[6] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m5, m7);
    v[7] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m5, m7);
}
