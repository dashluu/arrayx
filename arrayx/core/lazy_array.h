#pragma once

#include "exceptions.h"
#include "range.h"
#include "shape.h"
#include "dtype.h"
#include "id.h"
#include "../device/buffer.h"
#include "../device/device.h"

namespace ax::core
{
    class LazyArray : public std::enable_shared_from_this<LazyArray>, public IStr
    {
    private:
        static IdGenerator id_gen;
        Id id;
        Shape shape;
        DtypePtr dtype;
        using Buffer = ax::device::Buffer;
        using DevicePtr = ax::device::DevicePtr;
        DevicePtr device;

    public:
        std::shared_ptr<Buffer> buff = nullptr;

        LazyArray(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device)
        {
            buff = std::make_shared<Buffer>(ptr, nbytes);
        }

        LazyArray(const Shape &shape, DtypePtr dtype, DevicePtr device) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device)
        {
        }

        LazyArray(const LazyArray &arr) : id(id_gen.generate()), shape(arr.shape), dtype(arr.dtype), device(arr.device), buff(arr.buff)
        {
        }

        ~LazyArray() {}

        LazyArray &operator=(const LazyArray &arr) = delete;

        const Id &get_id() const { return id; }

        const Shape &get_shape() const { return shape; }

        isize get_offset() const { return shape.get_offset(); }

        const ShapeView &get_view() const { return shape.get_view(); }

        const ShapeStride &get_stride() const { return shape.get_stride(); }

        uint8_t *get_ptr() const { return buff->get_ptr() + get_offset() * get_itemsize(); }

        DtypePtr get_dtype() const { return dtype; }

        DevicePtr get_device() const { return device; }

        const std::string get_device_name() const { return device->get_name(); }

        isize get_numel() const { return shape.get_numel(); }

        isize get_ndim() const { return shape.get_ndim(); }

        isize get_itemsize() const { return dtype->get_size(); }

        isize get_nbytes() const { return get_numel() * get_itemsize(); }

        static LazyArrayPtr empty(const Shape &shape, DtypePtr dtype, DevicePtr device)
        {
            return std::make_shared<LazyArray>(shape, dtype, device);
        }

        static LazyArrayPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device)
        {
            return std::make_shared<LazyArray>(ptr, nbytes, shape, dtype, device);
        }

        bool is_contiguous() const { return shape.is_contiguous(); }

        // TODO: handle more cases to reduce copying?
        bool copy_when_reshape(const ShapeView &view) { return !is_contiguous(); }

        uint8_t *strided_elm_ptr(isize k) const;

        const std::string str() const override;
    };

    inline IdGenerator LazyArray::id_gen = IdGenerator();
    using LazyArrayPtrVec = std::vector<LazyArrayPtr>;
};