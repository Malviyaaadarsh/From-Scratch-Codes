"""
Summary of Notes : 

Tensor : Muiltidimensional array with uniform type,defined shape,stride and operations.
Rank of Tensor : No. of Axes. 
Shape : tuple listing size along each axis.
Stride : No. of elements to skip to advance one position along each axes.
Broadcasting : Strict ruleset. Align from right, dimension must be equal or one must be 1.
Einsum : expresses tensor contraction,outer product,trace or transpose in a line.
Contraction : Shared index b//w tensors is multiplied and summed,producing lower rank result.
View : tensor sharing same memory buffer but with diff. shape/stride metdata.
NCHW : (Batch,Channel,Height,Width) memory layout used by pytorch for image tensors.
NHWC : (Batch,Height,Width,Channel) layout used by tensorflow for image tensors.
"""

import numpy as np
from functools import reduce
from itertools import product as iterproduct


class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self.flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda x, y: x * y, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Data size {len(self._data)} does not match shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self.compute_strides(self._shape)

    def flatten_nested(self, data):
        if not isinstance(data, (list, tuple)):
            return [data], ()
        if not len(data):
            return [], (0,)

        i_res = [self.flatten_nested(i) for i in data]
        i_shape = i_res[0][1]
        for i, (_, s) in enumerate(i_res):
            if s != i_shape:
                raise ValueError(
                    f"Nested data has inconsistent shapes: {i_shape} vs {s} , at {i}"
                )

        flat = []
        for i_data, _ in i_res:
            flat.extend(i_data)
        return flat, (len(data),) + i_shape

    _flatten_nested = flatten_nested

    @staticmethod
    def compute_strides(shape):
        strides = [1] * len(shape)
        if len(shape) > 0:
            for i in range(len(shape) - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
        else:
            return ()
        return tuple(strides)

    _compute_strides = compute_strides

    @property
    def data(self):
        return self._data

    @property
    def rank(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def size(self):
        return len(self._data)

    def __add__(self, other):
        return self.elementwise_operation(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.elementwise_operation(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self.elementwise_operation(other, lambda a, b: a * b)

    def elementwise_operation(self, other, op):
        if isinstance(other, (int, float)):
            return Tensor([op(x, other) for x in self._data], self._shape)

        if not isinstance(other, Tensor):
            raise TypeError(
                f"Unsupported type for elementwise operation: {type(other)}"
            )

        if self._shape != other._shape:
            raise ValueError(
                f"Shapes {self._shape} and {other._shape} are not compatible for elementwise operation"
            )

        return Tensor([op(x, y) for x, y in zip(self._data, other._data)], self._shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.to_list()})"

    def to_numpy(self):
        return np.array(self._data).reshape(self._shape)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != self.rank:
            raise IndexError(f"Expected {self.rank} indices, got {len(indices)}")
        return self._data[self.flat_index(indices)]

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != self.rank:
            raise IndexError(f"Expected {self.rank} indices, got {len(indices)}")
        self._data[self.flat_index(indices)] = value

    def flat_index(self, indices):
        if len(indices) != self.rank:
            raise IndexError(f"Expected {self.rank} indices, got {len(indices)}")
        index = 0
        for axis, (i, stride) in enumerate(zip(indices, self._strides)):
            if i < 0 or i >= self._shape[axis]:
                raise IndexError(
                    f"Index {i} out of bounds for axis {axis} with size {self._shape[axis]}"
                )
            index += i * stride
        return index

    _flat_index = flat_index

    def squeeze(self, dim=None):
        if dim is not None:
            if self._shape[dim] != 1:
                return self.reshape(self._shape)
            shape = list(self._shape)
            shape.pop(dim)
            return self.reshape(tuple(shape) if shape else ())
        shape = tuple(s for s in self._shape if s != 1)
        return self.reshape(shape if shape else ())

    def unsqueeze(self, dim):
        if dim < 0:
            dim = len(self._shape) + dim + 1
        shape = list(self._shape)
        shape.insert(dim, 1)
        return self.reshape(tuple(shape))

    def reshape(self, new_shape):
        new_shape = list(new_shape)
        neg_i = -1
        known_product = 1

        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_i != -1:
                    raise ValueError("Only one dimension can be -1")
                neg_i = i
            else:
                known_product *= s

        if neg_i != -1:
            total_size = self.size
            if total_size % known_product != 0:
                raise ValueError(
                    f"Cannot reshape tensor of size {total_size} into shape {new_shape}"
                )
            new_shape[neg_i] = total_size // known_product

        total_size = reduce(lambda x, y: x * y, new_shape, 1)
        if total_size != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} into shape {new_shape}"
            )

        result = Tensor.__new__(Tensor)
        result._data = self._data[:]
        result._shape = tuple(new_shape)
        result._strides = self.compute_strides(result._shape)
        return result

    def transpose(self, dim1, dim2):
        dims = list(range(self.rank))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(dims)

    def permute(self, dims):
        if sorted(dims) != list(range(self.rank)):
            raise ValueError(
                f"Invalid permutation {dims} for tensor of rank {self.rank}"
            )

        new_shape = tuple(self._shape[d] for d in dims)
        result = Tensor.__new__(Tensor)
        result._data = [0] * self.size
        result._shape = new_shape
        result._strides = self.compute_strides(new_shape)

        for idx in iterproduct(*(range(s) for s in self._shape)):
            new_idx = tuple(idx[d] for d in dims)
            flat = sum(i * s for i, s in zip(idx, self._strides))
            new_flat = sum(i * s for i, s in zip(new_idx, result._strides))
            result._data[new_flat] = self._data[flat]

        return result

    def flatten(self, start_dim=0, end_dim=-1):
        if self.rank == 0:
            return self.reshape((1,))

        if start_dim < 0:
            start_dim += self.rank
        if end_dim < 0:
            end_dim += self.rank

        if start_dim < 0 or end_dim < 0 or start_dim >= self.rank or end_dim >= self.rank:
            raise IndexError("flatten dimensions out of range")
        if start_dim > end_dim:
            raise ValueError("start_dim must be <= end_dim")

        new_shape = list(self._shape[:start_dim])
        middle = self._shape[start_dim : end_dim + 1]
        middle_size = reduce(lambda a, b: a * b, middle, 1)
        new_shape.append(middle_size)
        new_shape.extend(self._shape[end_dim + 1 :])
        return self.reshape(tuple(new_shape))

    def sum(self, axis=None):
        if axis is None:
            return sum(self._data)

        if axis < 0:
            axis += self.rank
        if axis < 0 or axis >= self.rank:
            raise IndexError(f"Axis {axis} out of range for rank {self.rank}")

        new_shape = list(self._shape)
        new_shape.pop(axis)
        result_size = reduce(lambda x, y: x * y, new_shape, 1)
        result_data = [0] * result_size
        result_strides = self.compute_strides(tuple(new_shape))

        for idx in iterproduct(*(range(s) for s in self._shape)):
            flat = sum(i * s for i, s in zip(idx, self._strides))
            reduced_idx = idx[:axis] + idx[axis + 1 :]
            new_flat = sum(i * s for i, s in zip(reduced_idx, result_strides)) if reduced_idx else 0
            result_data[new_flat] += self._data[flat]

        return result_data[0] if not new_shape else Tensor(result_data, tuple(new_shape))

    def to_list(self):
        if self.rank == 0:
            return self._data[0]
        return self._build_nested(self._data, self._shape, 0)

    def _build_nested(self, data, shape, offset):
        if len(shape) == 1:
            return data[offset : offset + shape[0]]
        result = []
        stride = reduce(lambda a, b: a * b, shape[1:], 1)
        for i in range(shape[0]):
            result.append(self._build_nested(data, shape[1:], offset + i * stride))
        return result


if __name__ == "__main__":
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    print(t)
    print(t.shape, t.rank, t.size, t.strides)
    print(t[1, 2])
    print(t.reshape((3, 2)))
    print(t.sum(axis=0))