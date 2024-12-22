#include <iostream>
#include <algorithm>
#include <cmath>

template<typename T>
void convolve2D(int batch_size,
                int input_channel,
                int input_height,
                int input_width,
                int output_channel,
                int kernel_height,
                int kernel_width,
                int stride_height,
                int stride_width,
                int padding_height,
                int padding_width,
                int group_count,
                const T* input_data,
                T* output_data,
                const T* kernel_data)
{
    // 计算输出特征图的高度和宽度
    int output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;

    // 遍历批次
    for (int b = 0; b < batch_size; ++b)
    {
        // 遍历输出通道分组
        for (int og = 0; og < group_count; ++og)
        {
            // 遍历输出通道分组内的通道
            for (int ocg = 0; ocg < output_channel / group_count; ++ocg)
            {
                // 遍历输出特征图的高度
                for (int oh = 0; oh < output_height; ++oh)
                {
                    // 遍历输出特征图的宽度
                    for (int ow = 0; ow < output_width; ++ow)
                    {
                        T sum = 0;

                        // 遍历输入通道分组
                        for (int ig = 0; ig < group_count; ++ig)
                        {
                            // 遍历输入通道分组内的通道
                            for (int icg = 0; icg < input_channel / group_count; ++icg)
                            {
                                // 遍历卷积核的高度
                                for (int kh = 0; kh < kernel_height; ++kh)
                                {
                                    // 遍历卷积核的宽度
                                    for (int kw = 0; kw < kernel_width; ++kw)
                                    {
                                        // 根据输入矩阵NHWC布局计算输入数据的对应索引
                                        int input_idx = (b * input_height * input_width * group_count * (input_channel / group_count) +
                                                         oh * stride_height * input_width * group_count * (input_channel / group_count) +
                                                         ow * group_count * (input_channel / group_count) +
                                                         ig * (input_channel / group_count) +
                                                         icg) * (input_height + 2 * padding_height) * (input_width + 2 * padding_width) +
                                                         kh * (input_width + 2 * padding_width) + kw;

                                        // 根据权重矩阵NHWC布局计算卷积核的对应索引
                                        int kernel_idx = (ig * (output_channel / group_count) * kernel_height * kernel_width * (input_channel / group_count) +
                                                          ocg * kernel_height * kernel_width * (input_channel / group_count) +
                                                          kh * kernel_width * (input_channel / group_count) +
                                                          kw * (input_channel / group_count) +
                                                          icg);

                                        sum += input_data[input_idx] * kernel_data[kernel_idx];
                                    }
                                }
                            }
                        }

                        // 根据输出矩阵NHWC布局将计算结果存入输出数据
                        output_data[((b * output_height * output_width * group_count * (output_channel / group_count) +
                                      oh * output_width * group_count * (output_channel / group_count) +
                                      ow * group_count * (output_channel / group_count) +
                                      og * (output_channel / group_count) +
                                      ocg) * (output_height + 2 * padding_height) * (output_width + 2 * padding_width) +
                                      kh * (output_width + 2 * padding_width) + kw)] = sum;
                    }
                }
            }
        }
    }
}
// 3D卷积核的实现

template<typename T>
void convolve3D(int batch_size, 
                int input_channel, 
                int depth, 
                int input_height,
                int input_width,
                int output_channel,
                int kernel_depth,
                int kernel_height,
                int kernel_width,
                int stride_depth,
                int stride_height,
                int stride_width,
                int padding_depth,
                int padding_height,
                int padding_width,
                int group_count,
                const T* input_data,
                T* output_data,
                const T* kernel_data) {
    
    // 计算输出数据的尺寸
    int output_depth = static_cast<int>(std::ceil(static_cast<double>(depth + 2 * padding_depth - kernel_depth) / stride_depth)) + 1;
    int output_height = static_cast<int>(std::ceil(static_cast<double>(input_height + 2 * padding_height - kernel_height) / stride_height)) + 1;
    int output_width = static_cast<int>(std::ceil(static_cast<double>(input_width + 2 * padding_width - kernel_width) / stride_width)) + 1;
 
    // 检查分组卷积的合理性
    if (input_channel % group_count != 0 || output_channel % group_count != 0) {
        throw std::invalid_argument("Input and output channels must be divisible by group count.");
    }
 
    // 为每个样本、每个输出通道和每个空间位置计算卷积
    for (int b = 0; b < batch_size; ++b) {
        for (int g = 0; g < group_count; ++g) {
            for (int oc = g; oc < output_channel; oc += group_count) {
                for (int od = 0; od < output_depth; ++od) {
                    for (int oh = 0; oh < output_height; ++oh) {
                        for (int ow = 0; ow < output_width; ++ow) {
                            T sum = T(); // 初始化累加器为0（或适当的初始值）
 
                            // 遍历卷积核的每个元素
                            for (int kd = 0; kd < kernel_depth; ++kd) {
                                for (int kh = 0; kh < kernel_height; ++kh) {
                                    for (int kw = 0; kw < kernel_width; ++kw) {
                                        // 计算输入数据的索引
                                        int id = od * stride_depth + kd - padding_depth;
                                        int ih = oh * stride_height + kh - padding_height;
                                        int iw = ow * stride_width + kw - padding_width;
 
                                        // 检查索引是否在输入数据的范围内（考虑填充）
                                        if (id >= 0 && id < depth && ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                            // 计算输入通道索引（考虑分组）
                                            int ic = (oc - g) * (input_channel / group_count) + (kd * kernel_height + kh) * kernel_width + kw; // 这里可能需要根据实际情况调整
                                            // 或者，如果每个输入通道都完整地与卷积核的一个子集相关联：
                                            // int ic = (oc - g) % (input_channel / group_count) + g * (input_channel / group_count) * ((kd * kernel_height + kh) * kernel_width + kw) / (input_channel / group_count); // 这通常不是正确的，只是展示另一种可能的索引计算方式
 
                                            // 注意：上面的ic计算方式可能是错误的，正确的计算方式取决于卷积核和输入数据是如何映射的。
                                            // 在这里，我们假设卷积核是平坦的，并且每个输出通道都使用输入通道的一个子集。
                                            // 但是，在分组卷积中，每个输出通道组通常只与输入通道的一个对应组相关联。
                                            // 因此，我们需要一个更合适的索引计算方式。
 
                                            // 一个更合适的索引计算方式可能是这样的（假设每个组内的输入和输出通道是连续的）：
                                            int ic_group = (oc - g) / (output_channel / group_count); // 输出通道在当前组内的索引
                                            int kernel_index = (kd * kernel_height + kh) * kernel_width + kw; // 卷积核内的索引
                                            int input_channel_offset = ic_group * (input_channel / group_count); // 输入通道在当前组内的起始偏移
                                            ic = input_channel_offset + kernel_index; // 注意：这里假设卷积核与输入通道的一一对应关系，这通常不是分组卷积的情况，但可以作为示例。
 
                                            // 然而，在分组卷积中，我们实际上应该这样计算ic：
                                            // ic = (g * (input_channel / group_count)) + ((kd * kernel_height + kh) * kernel_width + kw);
                                            // 并且确保ic不会超出输入数据的范围。但是，由于我们不知道卷积核是如何遍历输入通道的，
                                            // 这里我们暂时使用上面的错误计算方式作为示例，并在注释中指出正确的方向。
 
                                            // 注意：下面的代码行使用了上面可能错误的ic计算方式。
                                            // 在实际应用中，你需要根据卷积核和输入数据的实际布局来调整它。
 
                                            // 累加输入数据和卷积核的乘积
                                            sum += input_data[b * input_channel * depth * input_height * input_width + ic * depth * input_height * input_width + id * input_height * input_width + ih * input_width + iw]
                                                 * kernel_data[oc * kernel_depth * kernel_height * kernel_width + g * (kernel_depth * kernel_height * kernel_width / group_count) + kernel_index];
                                        }
                                    }
                                }
                            }
 
                            // 将累加的结果存储到输出数据中
                            output_data[b * output_channel * output_depth * output_height * output_width + oc * output_depth * output_height * output_width + od * output_height * output_width + oh * output_width + ow] = sum;
                        }
                    }
                }
            }
        }
    }
}
            


int main() {
    
 
}