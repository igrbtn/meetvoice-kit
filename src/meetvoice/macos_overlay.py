"""macOS floating translator window: a small translucent HUD panel over every app.

A non-activating NSPanel (clicks do not steal focus from the meeting app), on all Spaces
and above full-screen apps, holding a WKWebView with the overlay page.
sharingType=None keeps it out of screen sharing and screenshots.

Main thread only; needs a running NSApplication (rumps, PyObjC app, or `run_standalone`).
Requires: pyobjc-framework-WebKit (extra "macos").
"""

WIDTH, HEIGHT, MARGIN = 440, 280, 24
ALPHA = 0.9
AUTOSAVE = "MeetvoiceOverlay"


class Overlay:
    def __init__(self, url: str, title: str = "Live translation"):
        self.url, self.title = url, title
        self.panel = None
        self.web = None

    def _build(self):
        import AppKit
        from Foundation import NSURL, NSURLRequest
        from WebKit import WKWebView, WKWebViewConfiguration

        screen = AppKit.NSScreen.mainScreen().visibleFrame()
        rect = AppKit.NSMakeRect(screen.origin.x + screen.size.width - WIDTH - MARGIN,
                                 screen.origin.y + MARGIN, WIDTH, HEIGHT)
        mask = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskResizable | AppKit.NSWindowStyleMaskUtilityWindow
                | AppKit.NSWindowStyleMaskHUDWindow | AppKit.NSWindowStyleMaskNonactivatingPanel)
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, mask, AppKit.NSBackingStoreBuffered, False)
        panel.setTitle_(self.title)
        panel.setFloatingPanel_(True)
        panel.setLevel_(AppKit.NSStatusWindowLevel)  # above a full-screen meeting window too
        panel.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                     | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
        panel.setHidesOnDeactivate_(False)
        panel.setReleasedWhenClosed_(False)
        panel.setAlphaValue_(ALPHA)
        panel.setSharingType_(AppKit.NSWindowSharingNone)
        panel.setMinSize_(AppKit.NSMakeSize(260, 120))
        panel.setFrameUsingName_(AUTOSAVE)
        panel.setFrameAutosaveName_(AUTOSAVE)
        web = WKWebView.alloc().initWithFrame_configuration_(
            panel.contentView().bounds(), WKWebViewConfiguration.alloc().init())
        web.setValue_forKey_(False, "drawsBackground")
        web.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        panel.setContentView_(web)
        web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(self.url)))
        self.panel, self.web = panel, web

    @property
    def visible(self) -> bool:
        return bool(self.panel and self.panel.isVisible())

    def show(self) -> None:
        if self.panel is None:
            self._build()
        elif not self.visible:
            self.web.reload()
        self.panel.orderFrontRegardless()

    def hide(self) -> None:
        if self.panel is not None:
            self.panel.orderOut_(None)


def run_standalone(url: str) -> None:
    """Show the panel in a process of its own (accessory app, no Dock icon). Blocks."""
    import AppKit

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    overlay = Overlay(url)
    overlay.show()
    app.run()
