use v_frame::{pixel::Pixel, plane::Plane};

pub(crate) fn calculate_histogram_rust<T: Pixel>(plane: &Plane<T>, bit_depth: usize) -> Vec<f64> {
    let mut hist = vec![0.0; 1 << bit_depth];

    for pixel in plane.data().iter() {
        let val = pixel.to_u16().unwrap() as usize;
        if val < hist.len() {
            hist[val] += 1.0;
        }
    }

    let total_pixels = plane.width().get() * plane.height().get();
    let inv_total = 1.0 / total_pixels as f64;
    for bin in hist.iter_mut() {
        *bin *= inv_total;
    }

    hist
}
