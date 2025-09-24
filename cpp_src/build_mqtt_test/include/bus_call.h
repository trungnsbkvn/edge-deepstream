#pragma once

#include <gst/gst.h>
#include <glib.h>

namespace EdgeDeepStream {

// Bus message callback function
gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data);

} // namespace EdgeDeepStream