use core::arch::wasm32::*;

use v_frame::{pixel::Pixel, plane::Plane};

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn calculate_histogram_wasm<T: Pixel>(
    plane: &Plane<T>,
    bit_depth: usize,
) -> Vec<f64> {
    if std::mem::size_of::<T>() != 1 {
        return super::rust::calculate_histogram_rust(plane, bit_depth);
    }

    let w = plane.width().get();
    let h = plane.height().get();
    let stride = plane.geometry().stride.get();

    let mut h0 = [0u32; 256];
    let mut h1 = [0u32; 256];
    let mut h2 = [0u32; 256];
    let mut h3 = [0u32; 256];

    let ptr = plane.data().as_ptr() as *const u8;

    for y in 0..h {
        let row_ptr = unsafe { ptr.add(y * stride) };
        let mut x = 0;

        while x + 16 <= w {
            let v = unsafe { v128_load(row_ptr.add(x) as *const v128) };

            h0[u8x16_extract_lane::<0>(v) as usize] += 1;
            h1[u8x16_extract_lane::<1>(v) as usize] += 1;
            h2[u8x16_extract_lane::<2>(v) as usize] += 1;
            h3[u8x16_extract_lane::<3>(v) as usize] += 1;

            h0[u8x16_extract_lane::<4>(v) as usize] += 1;
            h1[u8x16_extract_lane::<5>(v) as usize] += 1;
            h2[u8x16_extract_lane::<6>(v) as usize] += 1;
            h3[u8x16_extract_lane::<7>(v) as usize] += 1;

            h0[u8x16_extract_lane::<8>(v) as usize] += 1;
            h1[u8x16_extract_lane::<9>(v) as usize] += 1;
            h2[u8x16_extract_lane::<10>(v) as usize] += 1;
            h3[u8x16_extract_lane::<11>(v) as usize] += 1;

            h0[u8x16_extract_lane::<12>(v) as usize] += 1;
            h1[u8x16_extract_lane::<13>(v) as usize] += 1;
            h2[u8x16_extract_lane::<14>(v) as usize] += 1;
            h3[u8x16_extract_lane::<15>(v) as usize] += 1;

            x += 16;
        }

        while x < w {
            let val = unsafe { *row_ptr.add(x) } as usize;
            h0[val] += 1;
            x += 1;
        }
    }

    let total_pixels = (w * h) as f64;
    let inv_total = 1.0 / total_pixels;

    let mut hist = vec![0.0; 1 << bit_depth];
    for i in 0..256 {
        let count = h0[i] + h1[i] + h2[i] + h3[i];
        if i < hist.len() {
            hist[i] = count as f64 * inv_total;
        }
    }

    hist
}
