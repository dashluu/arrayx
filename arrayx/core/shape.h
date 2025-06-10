#pragma once

#include "range.h"
#include <set>

namespace ax::core {
    using ShapeView = std::vector<isize>;
    using ShapeStride = std::vector<isize>;
    using ShapeDims = std::vector<isize>;

    class Shape {
    private:
        isize offset;
        ShapeView view;
        ShapeStride stride;

        void if_ranges_are_valid(const RangeVec &ranges) const {
            if (ranges.size() != get_ndim()) {
                throw std::invalid_argument("The number of ranges does not match the number of dimensions: " +
                                            std::to_string(ranges.size()) + " and " +
                                            std::to_string(get_ndim()) + ".");
            }

            for (size_t i = 0; i < ranges.size(); i++) {
                const Range range = ranges[i];
                if (range.start < 0 || range.start >= static_cast<isize>(view[i])) {
                    throw std::invalid_argument("Start " + std::to_string(range.start) + " is not in the range [0, " + std::to_string(view[i]) + ").");
                }
                if (range.stop < -1 || range.stop > static_cast<isize>(view[i])) {
                    throw std::invalid_argument("Stop " + std::to_string(range.stop) + " is not in the range [-1, " + std::to_string(view[i]) + "].");
                }
                if (range.step == 0) {
                    throw std::invalid_argument("Step cannot be zero.");
                }
                if (range.start < range.stop && range.step < 0) {
                    throw std::invalid_argument("Step " + std::to_string(range.step) + " is not positive when start " + std::to_string(range.start) + " < stop " + std::to_string(range.stop) + ".");
                }
                if (range.start > range.stop && range.step > 0) {
                    throw std::invalid_argument("Step " + std::to_string(range.step) + " is not negative when start " + std::to_string(range.start) + " > stop " + std::to_string(range.stop) + ".");
                }
            }
        }

        void if_dims_make_valid_permutation(const ShapeDims &dims) const {
            isize ndim = get_ndim();

            if (dims.size() != ndim) {
                throw std::invalid_argument("The number of dimensions in the specified order does not match the number of dimensions in the shape: " +
                                            std::to_string(dims.size()) + " and " + std::to_string(ndim) + ".");
            }

            std::vector<bool> dims_used(ndim, false);

            for (auto &dim : dims) {
                if (dim < 0 || dim >= ndim) {
                    throw std::invalid_argument("The dimension must be in the range [0, " + std::to_string(ndim) + ") but got " + std::to_string(dim) + ".");
                }
                dims_used[dim] = true;
            }

            for (auto dim_used : dims_used) {
                if (!dim_used) {
                    throw std::invalid_argument("The specified order must be a permutation of the dimensions but got " + vnumstr(dims) + ".");
                }
            }
        }

        void if_start_end_dim_are_valid(isize start_dim, isize end_dim) const {
            if (start_dim > end_dim) {
                throw std::invalid_argument("The start dimension must be smaller than the end dimension.");
            }
            if (start_dim < 0 || start_dim >= get_ndim()) {
                throw std::invalid_argument("The start dimension must be in the range [0, " + std::to_string(get_ndim()) + ") but got " + std::to_string(start_dim) + ".");
            }
            if (end_dim < 0 || end_dim >= get_ndim()) {
                throw std::invalid_argument("The end dimension must be in the range [0, " + std::to_string(get_ndim()) + ") but got " + std::to_string(end_dim) + ".");
            }
        }

    public:
        Shape() : Shape(0, {1}, {1}) {}

        Shape(isize offset, const ShapeView &view, const ShapeStride &stride) {
            if_view_is_valid(view);
            if (view.size() != stride.size()) {
                throw std::invalid_argument("View and stride do not have the same number of dimensions: " + std::to_string(view.size()) + " and " + std::to_string(stride.size()) + ".");
            }
            this->offset = offset;
            this->view = view;
            this->stride = stride;
        }

        Shape(isize offset, const ShapeView &view) {
            if_view_is_valid(view);
            this->offset = offset;
            this->view = view;
            stride.resize(view.size());
            isize s = 1;

            for (ssize_t i = view.size() - 1; i >= 0; i--) {
                stride[i] = s;
                s *= view[i];
            }
        }

        Shape(const ShapeView &view) : Shape(0, view) {}

        Shape(const Shape &shape) : Shape(shape.offset, shape.view, shape.stride) {}

        Shape &operator=(const Shape &shape) {
            offset = shape.offset;
            view = shape.view;
            stride = shape.stride;
            return *this;
        }

        isize get_offset() const { return offset; }

        const ShapeView &get_view() const { return view; }

        const ShapeStride &get_stride() const { return stride; }

        bool is_contiguous() const { return stride == get_contiguous_stride(); }

        static void if_view_is_valid(const ShapeView &view) {
            if (view.size() == 0) {
                throw std::invalid_argument("Shape must have at least one dimension.");
            }
            if (std::any_of(view.begin(), view.end(), [](isize v) { return v == 0; })) {
                throw std::invalid_argument("Dimension cannot be zero.");
            }
        }

        ShapeStride get_contiguous_stride() const {
            ShapeStride contiguous_stride(get_ndim(), 0);
            isize s = 1;

            for (isize i = get_ndim() - 1; i >= 0; i--) {
                contiguous_stride[i] = s;
                s *= view[i];
            }

            return contiguous_stride;
        }

        std::vector<isize> get_elms_per_dim() const {
            std::vector<isize> elms_per_dim(get_ndim(), 0);
            isize n = 1;

            for (isize i = get_ndim() - 1; i >= 0; i--) {
                n *= view[i];
                elms_per_dim[i] = n;
            }

            return elms_per_dim;
        }

        isize get_ndim() const { return view.size(); }

        isize get_numel() const { return std::accumulate(view.begin(), view.end(), 1, std::multiplies<isize>()); }

        bool broadcastable(const ShapeView &rhs) const {
            if (view == rhs) {
                return true;
            }

            for (auto view_iter = view.rbegin(), rhs_iter = rhs.rbegin();
                 view_iter != view.rend() && rhs_iter != rhs.rend();
                 view_iter++, rhs_iter++) {
                if (*view_iter != *rhs_iter && *view_iter != 1 && *rhs_iter != 1) {
                    return false;
                }
            }

            return true;
        }

        // One-direction broadcast check
        bool broadcastable_to(const ShapeView &target) const {
            if (view == target) {
                return true;
            }

            if (get_ndim() > target.size()) {
                return false;
            }

            for (auto view_iter = view.rbegin(), target_iter = target.rbegin(); view_iter != view.rend(); view_iter++, target_iter++) {
                if (*view_iter != *target_iter && *view_iter != 1) {
                    return false;
                }
            }

            return true;
        }

        bool matmul_broadcastable(const ShapeView &rhs) const {
            if (get_ndim() < 2 || view[get_ndim() - 1] != rhs[rhs.size() - 2]) {
                return false;
            }

            for (auto view_iter = view.begin(), rhs_iter = rhs.begin();
                 view_iter != view.end() - 2 && rhs_iter != rhs.end() - 2;
                 view_iter++, rhs_iter++) {
                if (*view_iter != *rhs_iter && *view_iter != 1 && *rhs_iter != 1) {
                    return false;
                }
            }

            return true;
        }

        // One-direction broadcast
        std::pair<Shape, ShapeDims> broadcast_to(const ShapeView &target) const {
            ShapeDims broadcast_dims;

            if (view == target) {
                return std::make_pair(*this, broadcast_dims);
            }

            if (!broadcastable_to(target)) {
                throw std::invalid_argument("Cannot broadcast shape (" + vnumstr(view) + ") to (" + vnumstr(target) + ").");
            }

            ShapeView broadcast_view = view;
            size_t ndim_diff = target.size() - broadcast_view.size();
            broadcast_view.insert(broadcast_view.begin(), ndim_diff, 1);
            Shape broadcast_shape(offset, broadcast_view);
            std::fill_n(broadcast_shape.stride.begin(), ndim_diff, 0);

            for (size_t i = 0; i < target.size(); i++) {
                if (broadcast_view[i] < target[i]) {
                    broadcast_dims.emplace_back(i);
                    broadcast_shape.view[i] = target[i];
                    broadcast_shape.stride[i] = 0;
                }
            }

            return std::make_pair(broadcast_shape, broadcast_dims);
        }

        // ShapeDims specifies which dimensions are broadcasted
        std::pair<Shape, ShapeDims> broadcast(const ShapeView &rhs) const {
            ShapeDims broadcast_dims;

            if (view == rhs) {
                return std::make_pair(*this, broadcast_dims);
            }

            if (!broadcastable(rhs)) {
                throw std::invalid_argument("Cannot broadcast shape (" + vnumstr(view) + ") and (" + vnumstr(rhs) + ").");
            }

            ShapeView lview = view;
            ShapeView rview = rhs;
            size_t ndim = std::max(lview.size(), rview.size());
            size_t ldiff = ndim - lview.size();
            size_t rdiff = ndim - rview.size();
            lview.insert(lview.begin(), ldiff, 1);
            rview.insert(rview.begin(), rdiff, 1);
            Shape broadcast_shape(offset, lview);
            std::fill_n(broadcast_shape.stride.begin(), ldiff, 0);

            for (size_t i = 0; i < ndim; i++) {
                if (lview[i] < rview[i]) {
                    broadcast_dims.emplace_back(i);
                    broadcast_shape.view[i] = rview[i];
                    broadcast_shape.stride[i] = 0;
                }
            }

            return std::make_pair(broadcast_shape, broadcast_dims);
        }

        Shape reshape(const ShapeView &target) const {
            // TODO: fix this
            if_view_is_valid(target);
            isize numel = get_numel();
            isize target_numel = std::accumulate(target.begin(), target.end(), 1, std::multiplies<isize>());
            if (numel != target_numel) {
                throw std::invalid_argument("Cannot reshape array of " + std::to_string(numel) + " to " + std::to_string(target_numel) + " elements.");
            }
            return Shape(offset, target);
        }

        ShapeDims transpose(isize start_dim, isize end_dim) const {
            if_start_end_dim_are_valid(start_dim, end_dim);
            ShapeDims transpose_dims(get_ndim());
            std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
            std::reverse(transpose_dims.begin() + start_dim, transpose_dims.begin() + end_dim + 1);
            return transpose_dims;
        }

        ShapeView flatten(isize start_dim, isize end_dim) const {
            if_start_end_dim_are_valid(start_dim, end_dim);
            ShapeView flattened_view = view;
            isize prod = std::accumulate(flattened_view.begin() + start_dim, flattened_view.begin() + end_dim + 1, 1, std::multiplies<isize>());
            // Erase from start_dim + 1 to end_dim + 1
            flattened_view.erase(flattened_view.begin() + start_dim + 1, flattened_view.begin() + end_dim + 1);
            // Update view at start_dim
            flattened_view[start_dim] = prod;
            return flattened_view;
        }

        Shape permute(const ShapeDims &dims) const {
            if_dims_make_valid_permutation(dims);
            isize ndim = get_ndim();
            ShapeView v(ndim, 0);
            ShapeStride s(ndim, 0);

            for (isize i = 0; i < ndim; i++) {
                v[i] = view[dims[i]];
                s[i] = stride[dims[i]];
            }

            return Shape(offset, v, s);
        }

        ShapeView undo_permute_view(const ShapeDims &dims) const {
            /*
            Example 1:
            2, 2, 4, 3, 5
            1, 2, 0, 4, 3
            2, 4, 2, 5, 3
            2, 0, 1, 4, 3
            2, 2, 4, 3, 5

            Example 2:
            2 3 4
            2 0 1
            4 2 3
            1 2 0
            2 3 4

            Example 3:
            2 3 4 5
            1 3 2 0
            3 5 4 2
            3 0 2 1
            2 3 4 5
            */
            if_dims_make_valid_permutation(dims);
            ShapeView reverse_dims(dims.size());

            for (size_t i = 0; i < dims.size(); i++) {
                reverse_dims[dims[i]] = i;
            }

            return reverse_dims;
        }

        Shape undo_permute(const ShapeDims &dims) const {
            return permute(undo_permute_view(dims));
        }

        Shape slice(const RangeVec &ranges) const {
            if_ranges_are_valid(ranges);
            isize o = offset;

            for (size_t i = 0; i < ranges.size(); i++) {
                o += ranges[i].start * stride[i];
            }

            ShapeView v(get_ndim());
            ShapeStride s(get_ndim());

            for (size_t i = 0; i < ranges.size(); i++) {
                const Range &range = ranges[i];
                isize diff = std::abs(range.stop - range.start);
                v[i] = static_cast<isize>(ceil((static_cast<double>(diff)) / std::abs(range.step)));
                s[i] = stride[i] * range.step;
            }

            return Shape(o, v, s);
        }

        Shape unsqueeze(const ShapeDims &dims) const {
            ShapeView v = view;
            ShapeStride s = stride;

            if (dims.empty()) {
                v.push_back(1);
                s.push_back(1);
                return Shape(offset, v, s);
            }

            // Check for duplicates
            // Set is red black tree under the hood, which can be used for both sorting and checking for uniqueness
            std::set<isize> unique_dims(dims.begin(), dims.end());
            if (unique_dims.size() != dims.size()) {
                throw std::invalid_argument("Duplicate dimensions in unsqueeze.");
            }

            // Check for invalid dimension
            for (auto &dim : unique_dims) {
                if (dim < 0 || dim > get_ndim()) {
                    throw std::invalid_argument("Dimension " + std::to_string(dim) + " is out of range [0, " + std::to_string(get_ndim()) + "] during unsqueeze.");
                }
            }

            // Process in descending order using reverse iterator
            for (auto iter = unique_dims.rbegin(); iter != unique_dims.rend(); ++iter) {
                v.insert(v.begin() + *iter, 1);
                s.insert(s.begin() + *iter, 0);
            }

            return Shape(offset, v, s);
        }

        Shape squeeze(const ShapeDims &dims) const {
            ShapeView v = view;
            ShapeStride s = stride;

            if (dims.empty()) {
                for (ssize_t i = view.size() - 1; i >= 0; i--) {
                    if (view[i] == 1) {
                        v.erase(v.begin() + i);
                        s.erase(s.begin() + i);
                    }
                }

                return Shape(offset, v, s);
            }

            // Check for duplicates
            std::set<isize> unique_dims(dims.begin(), dims.end());
            if (unique_dims.size() != dims.size()) {
                throw std::invalid_argument("Duplicate dimensions in squeeze.");
            }

            // Check for invalid dimension
            for (auto &dim : unique_dims) {
                if (dim < 0 || dim >= get_ndim()) {
                    throw std::invalid_argument("Dimension " + std::to_string(dim) + " is out of range [0, " + std::to_string(get_ndim()) + ") during squeeze.");
                }
                if (view[dim] != 1) {
                    throw std::invalid_argument("Dimension " + std::to_string(dim) + " is " + std::to_string(view[dim]) + " but not a singleton during squeeze.");
                }
            }

            // Process in descending order using reverse iterator
            for (auto iter = unique_dims.rbegin(); iter != unique_dims.rend(); ++iter) {
                v.erase(v.begin() + *iter);
                s.erase(s.begin() + *iter);
            }

            return Shape(offset, v, s);
        }

        bool operator==(const Shape &shape) const { return view == shape.view; }

        bool operator!=(const Shape &shape) const { return !(*this == shape); }

        isize operator[](isize dim) const { return view[dim]; }

        const std::string str() const {
            return "(offset: " + std::to_string(offset) + ", view: " + vnumstr(view) + ", stride: " + vnumstr(stride) + ")";
        }

        ShapeView::const_iterator cbegin() const {
            return view.cbegin();
        }

        ShapeView::const_iterator cend() const {
            return view.cend();
        }

        ShapeView::const_reverse_iterator crbegin() const {
            return view.crbegin();
        }

        ShapeView::const_reverse_iterator crend() const {
            return view.crend();
        }
    };
} // namespace ax::core

namespace std {
    template <>
    struct hash<ax::core::Shape> {
        std::size_t operator()(const ax::core::Shape &shape) const {
            std::size_t seed = 0;
            for (auto &v : shape.get_view()) {
                seed ^= std::hash<ax::core::isize>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
} // namespace std