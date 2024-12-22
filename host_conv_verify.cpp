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
    // �����������ͼ�ĸ߶ȺͿ��
    int output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;

    // ��������
    for (int b = 0; b < batch_size; ++b)
    {
        // �������ͨ������
        for (int og = 0; og < group_count; ++og)
        {
            // �������ͨ�������ڵ�ͨ��
            for (int ocg = 0; ocg < output_channel / group_count; ++ocg)
            {
                // �����������ͼ�ĸ߶�
                for (int oh = 0; oh < output_height; ++oh)
                {
                    // �����������ͼ�Ŀ��
                    for (int ow = 0; ow < output_width; ++ow)
                    {
                        T sum = 0;

                        // ��������ͨ������
                        for (int ig = 0; ig < group_count; ++ig)
                        {
                            // ��������ͨ�������ڵ�ͨ��
                            for (int icg = 0; icg < input_channel / group_count; ++icg)
                            {
                                // ��������˵ĸ߶�
                                for (int kh = 0; kh < kernel_height; ++kh)
                                {
                                    // ��������˵Ŀ��
                                    for (int kw = 0; kw < kernel_width; ++kw)
                                    {
                                        // �����������NHWC���ּ����������ݵĶ�Ӧ����
                                        int input_idx = (b * input_height * input_width * group_count * (input_channel / group_count) +
                                                         oh * stride_height * input_width * group_count * (input_channel / group_count) +
                                                         ow * group_count * (input_channel / group_count) +
                                                         ig * (input_channel / group_count) +
                                                         icg) * (input_height + 2 * padding_height) * (input_width + 2 * padding_width) +
                                                         kh * (input_width + 2 * padding_width) + kw;

                                        // ����Ȩ�ؾ���NHWC���ּ������˵Ķ�Ӧ����
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

                        // �����������NHWC���ֽ������������������
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
// 3D����˵�ʵ��

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
    
    // ����������ݵĳߴ�
    int output_depth = static_cast<int>(std::ceil(static_cast<double>(depth + 2 * padding_depth - kernel_depth) / stride_depth)) + 1;
    int output_height = static_cast<int>(std::ceil(static_cast<double>(input_height + 2 * padding_height - kernel_height) / stride_height)) + 1;
    int output_width = static_cast<int>(std::ceil(static_cast<double>(input_width + 2 * padding_width - kernel_width) / stride_width)) + 1;
 
    // ���������ĺ�����
    if (input_channel % group_count != 0 || output_channel % group_count != 0) {
        throw std::invalid_argument("Input and output channels must be divisible by group count.");
    }
 
    // Ϊÿ��������ÿ�����ͨ����ÿ���ռ�λ�ü�����
    for (int b = 0; b < batch_size; ++b) {
        for (int g = 0; g < group_count; ++g) {
            for (int oc = g; oc < output_channel; oc += group_count) {
                for (int od = 0; od < output_depth; ++od) {
                    for (int oh = 0; oh < output_height; ++oh) {
                        for (int ow = 0; ow < output_width; ++ow) {
                            T sum = T(); // ��ʼ���ۼ���Ϊ0�����ʵ��ĳ�ʼֵ��
 
                            // ��������˵�ÿ��Ԫ��
                            for (int kd = 0; kd < kernel_depth; ++kd) {
                                for (int kh = 0; kh < kernel_height; ++kh) {
                                    for (int kw = 0; kw < kernel_width; ++kw) {
                                        // �����������ݵ�����
                                        int id = od * stride_depth + kd - padding_depth;
                                        int ih = oh * stride_height + kh - padding_height;
                                        int iw = ow * stride_width + kw - padding_width;
 
                                        // ��������Ƿ����������ݵķ�Χ�ڣ�������䣩
                                        if (id >= 0 && id < depth && ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                            // ��������ͨ�����������Ƿ��飩
                                            int ic = (oc - g) * (input_channel / group_count) + (kd * kernel_height + kh) * kernel_width + kw; // ���������Ҫ����ʵ���������
                                            // ���ߣ����ÿ������ͨ���������������˵�һ���Ӽ��������
                                            // int ic = (oc - g) % (input_channel / group_count) + g * (input_channel / group_count) * ((kd * kernel_height + kh) * kernel_width + kw) / (input_channel / group_count); // ��ͨ��������ȷ�ģ�ֻ��չʾ��һ�ֿ��ܵ��������㷽ʽ
 
                                            // ע�⣺�����ic���㷽ʽ�����Ǵ���ģ���ȷ�ļ��㷽ʽȡ���ھ���˺��������������ӳ��ġ�
                                            // ��������Ǽ���������ƽ̹�ģ�����ÿ�����ͨ����ʹ������ͨ����һ���Ӽ���
                                            // ���ǣ��ڷ������У�ÿ�����ͨ����ͨ��ֻ������ͨ����һ����Ӧ���������
                                            // ��ˣ�������Ҫһ�������ʵ��������㷽ʽ��
 
                                            // һ�������ʵ��������㷽ʽ�����������ģ�����ÿ�����ڵ���������ͨ���������ģ���
                                            int ic_group = (oc - g) / (output_channel / group_count); // ���ͨ���ڵ�ǰ���ڵ�����
                                            int kernel_index = (kd * kernel_height + kh) * kernel_width + kw; // ������ڵ�����
                                            int input_channel_offset = ic_group * (input_channel / group_count); // ����ͨ���ڵ�ǰ���ڵ���ʼƫ��
                                            ic = input_channel_offset + kernel_index; // ע�⣺�����������������ͨ����һһ��Ӧ��ϵ����ͨ�����Ƿ��������������������Ϊʾ����
 
                                            // Ȼ�����ڷ������У�����ʵ����Ӧ����������ic��
                                            // ic = (g * (input_channel / group_count)) + ((kd * kernel_height + kh) * kernel_width + kw);
                                            // ����ȷ��ic���ᳬ���������ݵķ�Χ�����ǣ��������ǲ�֪�����������α�������ͨ���ģ�
                                            // ����������ʱʹ������Ĵ�����㷽ʽ��Ϊʾ��������ע����ָ����ȷ�ķ���
 
                                            // ע�⣺����Ĵ�����ʹ����������ܴ����ic���㷽ʽ��
                                            // ��ʵ��Ӧ���У�����Ҫ���ݾ���˺��������ݵ�ʵ�ʲ�������������
 
                                            // �ۼ��������ݺ;���˵ĳ˻�
                                            sum += input_data[b * input_channel * depth * input_height * input_width + ic * depth * input_height * input_width + id * input_height * input_width + ih * input_width + iw]
                                                 * kernel_data[oc * kernel_depth * kernel_height * kernel_width + g * (kernel_depth * kernel_height * kernel_width / group_count) + kernel_index];
                                        }
                                    }
                                }
                            }
 
                            // ���ۼӵĽ���洢�����������
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