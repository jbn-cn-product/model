use crate::Object;

const REG_MAX: usize = 16;

/// Compute Distribution Focal Loss (DFL)
/// Converts the distribution (16 reg values) into a single coordinate.
///
/// optimization:
/// - Avoids full softmax calculation if possible, or uses optimized softmax.
/// - Vectorized iteration layout.
///
/// dfl_data: slice of [16] floats for one coordinate (x1, y1, x2, or y2)
/// anchor_points_buffer: not needed if we just do implicit index weighting 0..15
pub fn dfl(dfl_data: &[f32]) -> f32 {
    // Softmax
    let mut sum_exp = 0.0;
    let mut max_val = f32::MIN;

    // Find max for numeric stability
    for &v in dfl_data {
        if v > max_val {
            max_val = v;
        }
    }

    // Compute exp sum
    // We can also compute the weighted sum in the same pass if we buffer exps
    let mut exps = [0.0; REG_MAX];
    for (i, &v) in dfl_data.iter().enumerate() {
        let e = (v - max_val).exp();
        exps[i] = e;
        sum_exp += e;
    }

    // Weighted Sum (0*p0 + 1*p1 + ... + 15*p15)
    let mut weighted_sum = 0.0;
    for (i, &e) in exps.iter().enumerate() {
        let prob = e / sum_exp;
        weighted_sum += prob * (i as f32);
    }

    weighted_sum
}

/// Decode a single anchor box from raw outputs
/// raw_ptr: pointer to start of 64 values (4 coords * 16 Reg) + 80 classes
/// stride: stride of the feature map (8, 16, 32)
/// anchor_x, anchor_y: grid coordinates
pub fn decode_box(
    reg_dist: &[f32], // 64 floats (16*4)
    anchor_x: f32,
    anchor_y: f32,
    stride: f32,
) -> [f32; 4] {
    // Layout: [left_dist(16), top_dist(16), right_dist(16), bottom_dist(16)]
    let lt = dfl(&reg_dist[0..16]);
    let tp = dfl(&reg_dist[16..32]);
    let rt = dfl(&reg_dist[32..48]);
    let bm = dfl(&reg_dist[48..64]);

    let x1 = (anchor_x - lt) * stride;
    let y1 = (anchor_y - tp) * stride;
    let x2 = (anchor_x + rt) * stride;
    let y2 = (anchor_y + bm) * stride;

    [x1, y1, x2, y2]
}

/// Optimized Non-Maximum Suppression (Hard NMS)
///
/// objects: List of detected objects (will be sorted in-place)
/// iou_threshold: 0.45 typical
///
/// Returns: list of indices to keep
pub fn non_max_suppression(objects: &mut [Object], iou_threshold: f32) -> Vec<usize> {
    if objects.is_empty() {
        return vec![];
    }

    // 1. Sort by confidence descending
    // Using stable sort to preserve order for equal confidences
    objects.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::with_capacity(objects.len());
    let mut suppressed = vec![false; objects.len()];

    for i in 0..objects.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);

        // Check intersection with remaining
        let rect_a = &objects[i].bbox;
        let area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1]);

        for j in (i + 1)..objects.len() {
            if suppressed[j] {
                continue;
            }

            let rect_b = &objects[j].bbox;

            // Intersection
            let x1 = rect_a[0].max(rect_b[0]);
            let y1 = rect_a[1].max(rect_b[1]);
            let x2 = rect_a[2].min(rect_b[2]);
            let y2 = rect_a[3].min(rect_b[3]);

            if x2 < x1 || y2 < y1 {
                continue;
            } // No overlap

            let inter_area = (x2 - x1) * (y2 - y1);
            let area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1]);

            let union_area = area_a + area_b - inter_area;
            let iou = inter_area / union_area;

            // Filter
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}
