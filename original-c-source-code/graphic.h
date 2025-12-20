#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>

extern Display *d;
extern Window w;
extern GC gc, gc2;
extern XEvent e;
extern unsigned long black, white;

extern init();
extern line(int x0, int y0, int x1, int y1);
extern line2(int x0, int y0, int x1, int y1);
extern pointset(int x, int y);
extern pointreset(int x, int y);
extern xpause();
extern flush();
extern box(int, int, int, int);

