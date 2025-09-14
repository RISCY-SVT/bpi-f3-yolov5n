#include "pipeline.hpp"
#include "types.hpp"
#include <cassert>
#include <iostream>

using namespace yolov5;

static ProcessedFrame make_pf(uint64_t id) {
    ProcessedFrame pf;
    pf.frame.frame_id = id;
    return pf;
}

int main() {
    FrameReorderer ro;

    // Case 1: hole at start (drop 0)
    ro.markDropped(0);
    ro.addFrame(make_pf(1));
    ProcessedFrame out;
    bool ok = ro.getNextFrame(out);
    assert(ok && out.frame.frame_id == 1);

    // Case 2: hole in middle
    ro.addFrame(make_pf(3));
    ro.addFrame(make_pf(2));
    ro.markDropped(4);

    ok = ro.getNextFrame(out);
    assert(ok && out.frame.frame_id == 2);
    ok = ro.getNextFrame(out);
    assert(ok && out.frame.frame_id == 3);

    // Case 3: hole at end (mark dropped next expected)
    ro.markDropped(5);
    ro.addFrame(make_pf(6));
    ok = ro.getNextFrame(out);
    assert(ok && out.frame.frame_id == 6);

    // Reset and mixed order
    ro.reset();
    ro.addFrame(make_pf(2));
    ro.addFrame(make_pf(0));
    ro.markDropped(1);
    ok = ro.getNextFrame(out);
    assert(ok && out.frame.frame_id == 0);
    ok = ro.getNextFrame(out);
    assert(ok && out.frame.frame_id == 2);

    ro.stop();
    std::cout << "test_reorderer: OK" << std::endl;
    return 0;
}

