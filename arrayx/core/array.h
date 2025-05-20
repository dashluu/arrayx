#pragma once

#include "exceptions.h"
#include "range.h"
#include "shape.h"
#include "dtype.h"
#include "device.h"
#include "buffer.h"
#include "id.h"

namespace ax::core
{
    class Array : public std::enable_shared_from_this<Array>, public IStr
    {
    private:
        static IdGenerator id_gen;
        Id id;
        Shape shape;
        DtypePtr dtype;
        DevicePtr device;
        std::shared_ptr<Buffer> buff = nullptr;

    public:
        Array(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype = &f32, DevicePtr device = &device0) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device)
        {
            buff = std::make_shared<Buffer>(ptr, nbytes, false);
        }

        Array(const Shape &shape, DtypePtr dtype = &f32, DevicePtr device = &device0) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device)
        {
        }

        Array(const Array &arr) : id(id_gen.generate()), shape(arr.shape), dtype(arr.dtype), device(arr.device), buff(arr.buff)
        {
        }

        ~Array()
        {
            if (buff != nullptr)
            {
                device->get_allocator()->free(buff);
            }
        }

        Array &operator=(const Array &arr) = delete;

        const Id &get_id() const { return id; }

        const Shape &get_shape() const { return shape; }

        isize get_offset() const { return shape.get_offset(); }

        const ShapeView &get_view() const { return shape.get_view(); }

        const ShapeStride &get_stride() const { return shape.get_stride(); }

        // Gets the buffer pointer without accounting for offset
        uint8_t *get_buff_ptr() const { return buff->get_ptr(); }

        // Gets the buffer pointer after accounting for offset
        uint8_t *get_ptr() const { return get_buff_ptr() + get_offset() * get_itemsize(); }

        DtypePtr get_dtype() const { return dtype; }

        DevicePtr get_device() const { return device; }

        std::shared_ptr<Buffer> get_buff() const { return buff; }

        isize get_numel() const { return shape.get_numel(); }

        isize get_ndim() const { return shape.get_ndim(); }

        isize get_itemsize() const { return dtype->get_size(); }

        isize get_nbytes() const { return get_numel() * get_itemsize(); }

        // Get the number of bytes of the buffer the array is working on
        // Note: get_buff_nbytes() != get_nbytes()
        isize get_buff_nbytes() const { return buff->get_nbytes(); }

        static ArrayPtr empty(const Shape &shape, DtypePtr dtype = &f32, DevicePtr device = &device0)
        {
            return std::make_shared<Array>(shape, dtype, device);
        }

        static ArrayPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype = &f32, DevicePtr device = &device0)
        {
            return std::make_shared<Array>(ptr, nbytes, shape, dtype, device);
        }

        // Only allocate if the buffer is null
        void alloc()
        {
            if (buff == nullptr)
            {
                buff = device->get_allocator()->alloc(get_nbytes());
            }
        }

        // Low overhead by pointing to another buffer
        void alloc(Buffer &buff)
        {
            if (this->buff == nullptr)
            {
                this->buff = std::make_shared<Buffer>(buff);
            }
        }

        bool is_contiguous() const { return shape.is_contiguous(); }

        // TODO: handle more cases to reduce copying?
        bool copy_when_reshape(const ShapeView &view) { return !is_contiguous(); }

        uint8_t *strided_elm_ptr(isize k) const;

        const std::string str() const override;
    };

    inline IdGenerator Array::id_gen = IdGenerator();
    using ArrayPtrVec = std::vector<ArrayPtr>;
    using ArrayPtrVecIter = ArrayPtrVec::iterator;
    using ArrayPtrVecConstIter = ArrayPtrVec::const_iterator;
};