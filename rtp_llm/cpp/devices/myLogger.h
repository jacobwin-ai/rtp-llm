#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

// 日志级别枚举
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// 获取日志级别字符串
inline const char* logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

// 基础日志打印函数
template<typename... Args>
void printLog(LogLevel level, const char* file, int line, Args&&... args) {
    // 获取当前时间（可选）
    // auto now = std::chrono::system_clock::now();
    // auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    
    // 输出日志级别
    oss << "[" << logLevelToString(level) << "] ";
    
    // 输出文件名和行号
    oss << "[" << file << ":" << line << "] ";
    
    // 输出日志内容
    ((oss << args), ...);
    
    // 输出到标准输出或标准错误
    if (level == LogLevel::ERROR) {
        std::cerr << oss.str() << std::endl;
    } else {
        std::cout << oss.str() << std::endl;
    }
}

// 方便使用的宏定义
#define LOG_DEBUG(...)   printLog(LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)    printLog(LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) printLog(LogLevel::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...)   printLog(LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)

// 如果你想保持与代码中现有 RTP_LLM_LOG_INFO 风格一致，可以定义：
#define CUSTOM_LOG_INFO(...) printLog(LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define CUSTOM_LOG_ERROR(...) printLog(LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)

template<typename TensorAccessor>
std::pair<double, double> calculateMyTensorSum(TensorAccessor&& accessor, size_t dim) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double value = accessor(i).template item<double>();
        sum1 += value;
        sum2 += value * value;
    }
    return {sum1, sum2};
}

inline void printMyBuffer1d(const std::string&  hint,
                            torch::Tensor&      tensor,
                            std::vector<size_t> dims,
                            size_t              column_start,
                            size_t              column_end,
                            size_t              max_print_lines,
                            bool                showDetail = true) {
    size_t dim1 = dims[0];
    LOG_INFO("buffer: ", hint.c_str(), " shapes: [", dim1, "], type:", tensor.options().dtype().name().data());
    std::stringstream ss;
    if (showDetail) {
        ss << "Buffer " << hint << " : ";
        auto   print_func = [&](size_t column_start, size_t column_end) {
            for (int i = column_start; i < column_end && i < dim1; i++) {
                double value = tensor[i].item<double>();
                ss << " i = " << i << " value = " << value;
            }
        };
        const auto [sum1, sum2] = calculateMyTensorSum(
            [&](size_t i) -> auto { return tensor[i]; }, // 访问器
            dim1          // 当前维度长度
        );
        print_func(column_start, column_end);
        ss << " ...... ";
        print_func(std::max((size_t)0, dim1 - (column_end - column_start)), dim1);

        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
        LOG_INFO(ss.str());
    }
}

inline void printMyBuffer2d(const std::string&  hint,
                   torch::Tensor&      tensor,
                   std::vector<size_t> dims,
                   size_t              column_start,
                   size_t              column_end,
                   size_t              max_print_lines,
                   bool                showDetail = true) {
    size_t dim1 = dims[0];
    size_t dim2 = dims[1];
    LOG_INFO("buffer: ", hint.c_str(), " shapes: [", dim1, ",", dim2, "], type:", tensor.options().dtype().name().data());
    size_t line_num = 0;
    if (showDetail) {
        for (int i = 0; i < dim1; i++) {
            std::stringstream ss;
            ss << "Buffer " << hint << " : ";
            ss << "[" << i << "]";
            auto   print_func = [&](size_t column_start, size_t column_end) {
                for (int j = column_start; j < column_end && j < dim2; j++) {
                    double value = tensor[i][j].item<double>();
                    ss << " k = " << j << " value = " << value;
                }
            };
            print_func(column_start, column_end);
            ss << " ...... ";
            print_func(std::max((size_t)0, dim2 - (column_end - column_start)), dim2);
            const auto [sum1, sum2] = calculateMyTensorSum(
                [&](size_t j) -> auto { return tensor[i][j]; },
                dim2
            );
            ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
            LOG_INFO(ss.str());
            line_num++;
            if (line_num > max_print_lines) {
                return;
            }
        }
    }
}

inline void printMyBuffer3d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        bool                showDetail = true)
{
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t line_num = 0;
    LOG_INFO("buffer: ", hint.c_str(), " shapes: [", dim1, ",", dim2, ",", dim3,"], type:", tensor.options().dtype().name().data());
    if (showDetail) {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                std::stringstream ss;
                ss << "Buffer " << hint << " : ";
                ss << "[" << i << ", " << j << "]";
                auto   print_func = [&](size_t column_start, size_t column_end) {
                    for (int k = column_start; k < column_end && k < dim3; k++) {
                        double value = tensor[i][j][k].item<double>();
                        ss << " k = " << k << " value = " << value;
                    }
                };
                print_func(column_start, column_end);
                ss << " ...... ";
                print_func(std::max((size_t)0, dim3 - (column_end - column_start)), dim3);
                const auto [sum1, sum2] = calculateMyTensorSum(
                    [&](size_t k) -> auto { return tensor[i][j][k]; },
                    dim3
                );
                ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                LOG_INFO(ss.str());
                line_num++;
                if (line_num > max_print_lines) {
                    return;
                }
            }
        }
    }

}


inline void printMyBufferData_(const rtp_llm::Buffer& buffer, const std::string& hint, bool showDetail = true) {
    if (buffer.isQBuffer()) {
        const rtp_llm::QBuffer* q_buffer = &(reinterpret_cast<const rtp_llm::QBuffer&>(buffer));
        printMyBufferData_(q_buffer->kernel(), hint + "_kernel");
        printMyBufferData_(q_buffer->scales(), hint + "_scales");
        if (q_buffer->zeros().type()) {
            printMyBufferData_(q_buffer->zeros(), hint + "_zeros");
        }
        return;
    }
    torch::Tensor tensor = Buffer2torchTensor(buffer, false);


    std::vector<size_t> dims            = buffer.shape();
    size_t              column_start    = 0;
    size_t              column_end      = 20;
    size_t              max_print_lines = 3;
    if (dims.size() > 3) {
        LOG_INFO("print buffer size not supported");
    } else if (dims.size() == 3) {
        printMyBuffer3d(hint, tensor, dims, column_start, column_end, max_print_lines, showDetail);
    } else if (dims.size() == 2) {
        printMyBuffer2d(hint, tensor, dims, column_start, column_end, max_print_lines, showDetail);
    } else if (dims.size() == 1) {
        printMyBuffer1d(hint, tensor, dims, column_start, column_end, max_print_lines, showDetail);
    } else {
        LOG_INFO("print buffer size not supported");
    }
}

