use std::sync::Arc;

use v_frame::{frame::Frame, pixel::Pixel, plane::Plane};

use super::{SceneChangeDetector, ScenecutResult};

mod rust;

#[cfg(target_arch = "wasm32")]
mod wasm;

const HISTOGRAM_THRESHOLD: f64 = 0.15;

impl<T: Pixel> SceneChangeDetector<T> {
    pub(super) fn histogram_scenecut(
        &self,
        frame1: &Arc<Frame<T>>,
        frame2: &Arc<Frame<T>>,
    ) -> ScenecutResult {
        let delta = self.calculate_histogram_delta(&frame1.y_plane, &frame2.y_plane);

        ScenecutResult {
            threshold: HISTOGRAM_THRESHOLD,
            inter_cost: delta,
            imp_block_cost: delta,
            forward_adjusted_cost: delta,
            backward_adjusted_cost: delta,
        }
    }

    fn calculate_histogram_delta(&self, plane1: &Plane<T>, plane2: &Plane<T>) -> f64 {
        let hist1 = self.make_histogram(plane1);
        let hist2 = self.make_histogram(plane2);

        self.chi_square_distance(&hist1, &hist2)
    }

    fn make_histogram(&self, plane: &Plane<T>) -> Vec<f64> {
        #[cfg(target_arch = "wasm32")]
        {
            unsafe {
                return wasm::calculate_histogram_wasm(plane, self.bit_depth);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            return rust::calculate_histogram_rust(plane, self.bit_depth);
        }
    }

    fn chi_square_distance(&self, hist1: &[f64], hist2: &[f64]) -> f64 {
        let mut score = 0.0;
        for (h1, h2) in hist1.iter().zip(hist2.iter()) {
            let sum = h1 + h2;
            if sum > 0.0 {
                score += ((h1 - h2).powi(2)) / sum;
            }
        }
        0.5 * score
    }
}
