#include <torch/extension.h>
#include "api/ops/binary.h"

at::Tensor torchvulkan::add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) 
{    
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, 4, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, 0, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::add(a, b, alpha); });
}

at::Tensor torchvulkan::add_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) 
{
    if (self.scalar_type() == at::kBool) {
        if (alpha.to<bool>() == false) return self.clone();
        return binary_op_vulkan(self, other, alpha, 4, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
    }
    return binary_op_vulkan(self, other, alpha, 0, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::add(a, b, alpha); });
}

at::Tensor torchvulkan::subtract_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, 1, [alpha](const at::Tensor& a, const at::Tensor& b) { return at::sub(a, b, alpha); });
}

at::Tensor torchvulkan::subtract_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, 1, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::sub(a, b, alpha); });
}

at::Tensor torchvulkan::rsub_scalar_vulkan(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return binary_op_vulkan(self, other, alpha, 7, [alpha](const at::Tensor& a, const at::Scalar& b) { return at::rsub(a, b, alpha); });
}

at::Tensor torchvulkan::multiply_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, 1.0, 2, [](const at::Tensor& a, const at::Tensor& b) { return at::mul(a, b); });
}

at::Tensor torchvulkan::multiply_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, 1.0, 2, [](const at::Tensor& a, const at::Scalar& b) { return at::mul(a, b); });
}

at::Tensor torchvulkan::divide_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;
    
    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.cpu().to(c10::kFloat).to(self.device());
    if (at::isIntegralType(other.scalar_type(), /* includeBool = */ true)) other_vulkan = other.cpu().to(c10::kFloat).to(other.device());

    return binary_op_vulkan(self_vulkan, other_vulkan, 1.0, 3, [](const at::Tensor& a, const at::Tensor& b) { return at::div(a, b); });
}

at::Tensor torchvulkan::divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    at::Tensor self_vulkan = self;

    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.cpu().to(c10::kFloat).to(self.device());

    return binary_op_vulkan(self_vulkan, other, 1.0, 3, [](const at::Tensor& a, const at::Scalar& b) { return at::div(a, b); });
}

at::Tensor torchvulkan::maximum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, 1.0, 4, [](const at::Tensor& a, const at::Tensor& b) { return at::max(a, b); });
}

at::Tensor torchvulkan::minimum_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, 1.0, 5, [](const at::Tensor& a, const at::Tensor& b) { return at::min(a, b); });
}

at::Tensor torchvulkan::pow_vulkan(const at::Tensor& self, const at::Tensor& other) {
    return binary_op_vulkan(self, other, 1.0, 6, [](const at::Tensor& a, const at::Tensor& b) { return at::pow(a, b); });
}

at::Tensor torchvulkan::pow_tensor_scalar_vulkan(const at::Tensor& self, const at::Scalar& other) {
    return binary_op_vulkan(self, other, 1.0, 6, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(a, b); });
}

at::Tensor torchvulkan::pow_scalar_vulkan(const at::Scalar& self, const at::Tensor& other) {
    return binary_op_vulkan(other, self, 1.0, 9, [](const at::Tensor& a, const at::Scalar& b) { return at::pow(b, a); });
}

at::Tensor torchvulkan::atan2_vulkan(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor self_vulkan = self;
    at::Tensor other_vulkan = other;

    if (at::isIntegralType(self.scalar_type(), /* includeBool = */ true)) self_vulkan = self.cpu().to(c10::kFloat).to(self.device());
    if (at::isIntegralType(other.scalar_type(), /* includeBool = */ true)) other_vulkan = other.cpu().to(c10::kFloat).to(other.device());

    return binary_op_vulkan(self_vulkan, other_vulkan, 1.0, 10, [](const at::Tensor& a, const at::Tensor& b) { return at::atan2(a, b); });
}
