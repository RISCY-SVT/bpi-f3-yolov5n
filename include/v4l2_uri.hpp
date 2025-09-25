#ifndef V4L2_URI_HPP
#define V4L2_URI_HPP

#include <string>
#include <unordered_map>

namespace yolov5 {

/**
 * @brief Parsed representation of a V4L2 URI string passed to --src.
 */
struct V4L2UriOptions {
    bool is_v4l2 = false;                       //!< True when scheme == v4l2.
    bool auto_device = false;                   //!< True when device component is "auto".
    std::string device;                         //!< Device component ("auto" or /dev/videoX or by-id path).
    std::unordered_map<std::string, std::string> query; //!< Raw query key/value pairs.

    std::string fmt = "yuyv";                  //!< Requested pixel format (lowercase).
    bool fmt_specified = false;                 //!< True when fmt was present in query.

    int fps = 30;                               //!< Requested frames per second.
    bool fps_specified = false;                 //!< True when fps key provided.

    int width = 1280;                           //!< Requested capture width in pixels.
    bool width_specified = false;               //!< True when width key provided.

    int height = 720;                           //!< Requested capture height in pixels.
    bool height_specified = false;              //!< True when height key provided.

    int buffers = 3;                            //!< Requested kernel buffer count.
    bool buffers_specified = false;             //!< True when buffers key provided.

    int poll_ms = 50;                           //!< Poll timeout in milliseconds.
    bool poll_specified = false;                //!< True when poll_ms key provided.

    int ttl_ms = 0;                             //!< TTL override for live reorder (0 => unchanged).
    bool ttl_specified = false;                 //!< True when ttl_ms key provided.
};

/**
 * @brief Parse V4L2 URI into structured options with validation.
 * @param uri Input string such as "v4l2:auto?fmt=yuyv&fps=30".
 * @param error Optional buffer receiving validation failure message.
 * @return Filled options struct; options.is_v4l2=false when scheme is not v4l2.
 */
V4L2UriOptions parse_v4l2_uri(const std::string& uri, std::string* error = nullptr);

} // namespace yolov5

#endif // V4L2_URI_HPP
