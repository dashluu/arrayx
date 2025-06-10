#pragma once

#include "../device/buffer.h"
#include "../device/device.h"
#include "dtype.h"
#include "exceptions.h"
#include "id.h"
#include "range.h"
#include "shape.h"

namespace ax::core {
    class Lazy : public std::enable_shared_from_this<Lazy> {
    private:
        static IdGenerator id_gen;
        Id id;
        Shape shape;
        DtypePtr dtype;
        using Buffer = ax::device::Buffer;
        using DevicePtr = ax::device::DevicePtr;
        DevicePtr device;
        std::shared_ptr<Buffer> buff = nullptr;

    public:
        Lazy(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device) {
            buff = std::make_shared<Buffer>(ptr, nbytes);
        }

        Lazy(const Shape &shape, DtypePtr dtype, DevicePtr device) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device) {}
        Lazy(const Lazy &lazy) : id(id_gen.generate()), shape(lazy.shape), dtype(lazy.dtype), device(lazy.device), buff(lazy.buff) {}
        ~Lazy() {}

        Lazy &operator=(const Lazy &lazy) = delete;
        const Id &get_id() const { return id; }
        const Shape &get_shape() const { return shape; }
        isize get_offset() const { return shape.get_offset(); }
        const ShapeView &get_view() const { return shape.get_view(); }
        const ShapeStride &get_stride() const { return shape.get_stride(); }
        std::shared_ptr<Buffer> get_buff() const { return buff; }

        void init_buff(std::shared_ptr<Buffer> buff) {
            if (this->buff == nullptr) {
                this->buff = buff;
            }
        }

        // Gets the buffer pointer without accounting for offset
        uint8_t *get_buff_ptr() const { return buff->get_ptr(); }
        // Gets the buffer pointer after accounting for offset
        uint8_t *get_ptr() const { return buff->get_ptr() + get_offset() * get_itemsize(); }
        DtypePtr get_dtype() const { return dtype; }
        DevicePtr get_device() const { return device; }
        const std::string get_device_name() const { return device->get_name(); }
        isize get_numel() const { return shape.get_numel(); }
        isize get_ndim() const { return shape.get_ndim(); }
        isize get_itemsize() const { return dtype->get_size(); }
        // Get the number of bytes of the buffer the array is working on
        // Note: get_buff_nbytes() != get_nbytes()
        isize get_buff_nbytes() const { return buff->get_nbytes(); }
        isize get_nbytes() const { return get_numel() * get_itemsize(); }

        static LazyPtr empty(const Shape &shape, DtypePtr dtype, DevicePtr device) {
            return std::make_shared<Lazy>(shape, dtype, device);
        }

        static LazyPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device) {
            return std::make_shared<Lazy>(ptr, nbytes, shape, dtype, device);
        }

        bool is_contiguous() const { return shape.is_contiguous(); }
        // TODO: handle more cases to reduce copying?
        bool copy_when_reshape(const ShapeView &view) { return !is_contiguous(); }
        uint8_t *strided_elm_ptr(isize k) const;
        const std::string str() const;
    };

    inline IdGenerator Lazy::id_gen = IdGenerator();
    using LazyPtrVec = std::vector<LazyPtr>;
}; // namespace ax::core