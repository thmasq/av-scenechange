use core::arch::wasm32::*;

use v_frame::{pixel::Pixel, plane::Plane};

use crate::data::plane::PlaneRegion;

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn sad_plane_internal<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
    if std::mem::size_of::<T>() != 1 {
        return super::rust::sad_plane_internal(src, dst);
    }

    let w = src.width().get();
    let h = src.height().get();

    let n_rows = h.min(dst.height().get());
    let n_cols = w.min(dst.width().get());

    let stride1 = src.geometry().stride.get();
    let stride2 = dst.geometry().stride.get();

    // SAFETY: We verified size_of::<T>() == 1, so casting to u8 pointers is valid.
    let ptr1 = src.data().as_ptr() as *const u8;
    let ptr2 = dst.data().as_ptr() as *const u8;

    let mut sum_u64 = 0u64;

    for y in 0..n_rows {
        let row1 = ptr1.add(y * stride1);
        let row2 = ptr2.add(y * stride2);
        let mut x = 0;

        let mut row_sum = u32x4_splat(0);
        while x + 16 <= n_cols {
            let a = v128_load(row1.add(x) as *const v128);
            let b = v128_load(row2.add(x) as *const v128);

            let diff = v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a));

            let diff_lo = u16x8_extend_low_u8x16(diff);
            let diff_hi = u16x8_extend_high_u8x16(diff);
            let sum_lo = i32x4_extadd_pairwise_i16x8(diff_lo as v128);
            let sum_hi = i32x4_extadd_pairwise_i16x8(diff_hi as v128);

            row_sum = i32x4_add(row_sum, i32x4_add(sum_lo, sum_hi));
            x += 16;
        }

        sum_u64 += (i32x4_extract_lane::<0>(row_sum)
            + i32x4_extract_lane::<1>(row_sum)
            + i32x4_extract_lane::<2>(row_sum)
            + i32x4_extract_lane::<3>(row_sum)) as u64;

        while x < n_cols {
            let a = *row1.add(x) as i32;
            let b = *row2.add(x) as i32;
            sum_u64 += (a - b).abs() as u64;
            x += 1;
        }
    }
    sum_u64
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn get_sad_internal<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    if std::mem::size_of::<T>() != 1 || w != 8 || h != 8 {
        return super::rust::get_sad_internal(src, dst, w, h, bit_depth);
    }

    let stride1 = src.plane_cfg.stride;
    let stride2 = dst.plane_cfg.stride;

    // SAFETY: We verified size_of::<T>() == 1.
    // PlaneRegion is indexed relative to the region, so we take the pointer to
    // (0,0).
    let ptr1 = src[0].as_ptr() as *const u8;
    let ptr2 = dst[0].as_ptr() as *const u8;

    let mut sum = u32x4_splat(0);

    for i in 0..8 {
        let r1 = v128_load64_zero(ptr1.add(i * stride1) as *const u64);
        let r2 = v128_load64_zero(ptr2.add(i * stride2) as *const u64);

        let diff = v128_or(u8x16_sub_sat(r1, r2), u8x16_sub_sat(r2, r1));

        let diff_16 = u16x8_extend_low_u8x16(diff);
        sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(diff_16 as v128));
    }

    (i32x4_extract_lane::<0>(sum)
        + i32x4_extract_lane::<1>(sum)
        + i32x4_extract_lane::<2>(sum)
        + i32x4_extract_lane::<3>(sum)) as u32
}
