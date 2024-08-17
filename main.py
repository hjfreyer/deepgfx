# -*- coding: utf-8 -*-
# Advanced zoom example. Like in Google Maps.
# It zooms only a tile, but not the whole image. So the zoomed tile occupies
# constant memory and not crams it with a huge resized image for the large zooms.
import random
import tkinter as tk
import tensorflow as tf
import threading
from tkinter import ttk
import datetime
import queue
from PIL import Image, ImageTk


WIDTH = 800
HEIGHT = 600


def scale_matrix(sx, sy):
    return tf.constant([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]])


def translation_matrix(dx, dy):
    return tf.constant([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])


# class AutoScrollbar(ttk.Scrollbar):
#     ''' A scrollbar that hides itself if it's not needed.
#         Works only if you use the grid geometry manager '''
#     def set(self, lo, hi):
#         if float(lo) <= 0.0 and float(hi) >= 1.0:
#             self.grid_remove()
#         else:
#             self.grid()
#             ttk.Scrollbar.set(self, lo, hi)

#     def pack(self, **kw):
#         raise tk.TclError('Cannot use pack with this widget')

#     def place(self, **kw):
#         raise tk.TclError('Cannot use place with this widget')


class Collapser:
    def __init__(self, q):
        self.q = q

    def get(self):
        req, *args = self.q.get()
        while q.qsize() and req == "render":
            req, *args = self.q.get()
        return req, *args


def worker(q):
    while True:
        req, *args = q.get()
        if req == "load_model":
            model = tf.saved_model.load("/tmp/tri.dgf")
        elif req == "render":

            transform, tx, ty, a, b, movement, cb = args

            start = datetime.datetime.now()
            print("Make inputs")
            X = tf.stack(
                tf.meshgrid(
                    tf.linspace(0.0, 1, WIDTH),
                    tf.linspace(0.0, 0.75, HEIGHT),
                    indexing="ij",
                ),
                axis=-1,
            )
            X = X[:, :, tf.newaxis, :]
            X = tf.concat([X, tf.ones_like(X[..., 0:1])], -1)
            X = transform @ tf.expand_dims(X, -1)
            X = tf.squeeze(X, -1)

            T = tf.constant([0], tf.float32)
            T = T[tf.newaxis, tf.newaxis, :]

            render_func = model.render.concrete_functions[0]
            render_func.variables[0].assign(tf.constant(tx, tf.float32))
            render_func.variables[1].assign(tf.constant(ty, tf.float32))
            render_func.variables[2].assign(tf.constant(a, tf.float32)),
            render_func.variables[3].assign(tf.constant(b, tf.float32)),
            render_func.variables[4].assign(tf.constant(movement, tf.float32)),
            # render_func.variables[5].assign(tf.constant(c1s, tf.float32)),
            # render_func.variables[6].assign(tf.constant(c1e, tf.float32)),
            # render_func.variables[7].assign(tf.constant(cp, tf.float32)),

            print("Make call render")
            img = model.render(
                X,
            )[:, :, 0, :]
            img = tf.broadcast_to(img, (WIDTH, HEIGHT, 4))
            img = tf.transpose(tf.cast(255 * img, tf.uint8), (1, 0, 2))
            print("Done: ", datetime.datetime.now() - start)
            cb(img)


class Zoom_Advanced(tk.Frame):
    """Advanced zoom of the image"""

    def __init__(self, mainframe, model: queue.Queue):
        """Initialize the main Frame"""
        tk.Frame.__init__(self, master=mainframe, width=WIDTH, height=HEIGHT)
        self.master.title("Zoom with mouse wheel")
        self.model = model

        self.master.columnconfigure(0, minsize=WIDTH)
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, minsize=HEIGHT)

        self.settings = tk.Frame(self.master, background="blue")
        self.settings.grid(row=0, column=1, sticky="NSEW")

        self.reload = tk.Button(
            self.settings, text="Reload", command=self.request_reload
        )
        self.reload.pack()

        self.tx = tk.Scale(
            self.settings,
            from_=0,
            to=2,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.tx.set(1)
        self.tx.pack()
        self.ty = tk.Scale(
            self.settings,
            from_=0,
            to=2,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.ty.set(1)
        self.ty.pack()
        self.a = tk.Scale(
            self.settings,
            from_=0,
            to=2,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.a.set(1)
        self.a.pack()
        self.b = tk.Scale(
            self.settings,
            from_=0,
            to=2,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.b.set(1)
        self.b.pack()
        self.movement = tk.Scale(
            self.settings,
            from_=0,
            to=1,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.movement.pack()
        self.circle1start = tk.Scale(
            self.settings,
            from_=0,
            to=1,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.circle1start.pack()
        self.circle1end = tk.Scale(
            self.settings,
            from_=0,
            to=1,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.circle1end.pack()
        self.cp = tk.Scale(
            self.settings,
            from_=0,
            to=1,
            resolution=0.05,
            command=lambda e: self.request_image(),
        )
        self.cp.pack()

        self.inbox = queue.Queue()

        # # Vertical and horizontal scrollbars for canvas
        # vbar = AutoScrollbar(self.master, orient='vertical')
        # hbar = AutoScrollbar(self.master, orient='horizontal')
        # vbar.grid(row=0, column=1, sticky='ns')
        # hbar.grid(row=1, column=0, sticky='we')
        # # Create canvas and put image on it
        # self.canvas = tk.Canvas(self.master, highlightthickness=0,
        #                         xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        self.canvas = tk.Label(self.master, text="Loading")
        # self.canvas = tk.Canvas(self.master, highlightthickness=0, width=WIDTH, height=HEIGHT)
        self.canvas.grid(row=0, column=0)
        # self.canvas.update()  # wait till canvas is created

        self.label = tk.Label(self.master, text="Hello!")
        self.label.grid(row=1, column=0)

        self.transform = tf.eye(3, dtype=tf.float32)

        # vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        # hbar.configure(command=self.scroll_x)
        # # Make the canvas expandable
        # self.master.rowconfigure(0, weight=1)
        # self.master.columnconfigure(0, weight=1)
        # # Bind events to the Canvas
        # self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind("<ButtonPress-1>", self.move_from)
        self.canvas.bind("<B1-Motion>", self.move_to)
        # self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind("<Button-5>", self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind("<Button-4>", self.wheel)  # only with Linux, wheel scroll up
        # self.image = IM  # open image
        self.width, self.height = WIDTH, HEIGHT  # self.image.size
        # self.imscale = 1.0  # scale for the canvaas image
        # self.delta = 1.3  # zoom magnitude
        # # Put image into container rectangle and use it to set proper coordinates to the image
        # self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        # # Plot some optional random rectangles for the test purposes
        # minsize, maxsize, number = 5, 20, 10
        # for n in range(number):
        #     x0 = random.randint(0, self.width - maxsize)
        #     y0 = random.randint(0, self.height - maxsize)
        #     x1 = x0 + random.randint(minsize, maxsize)
        #     y1 = y0 + random.randint(minsize, maxsize)
        #     color = ('red', 'orange', 'yellow', 'green', 'blue')[random.randint(0, 4)]
        #     self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, activefill='black')
        self.request_reload()
        self.check_queue()

    def request_reload(self):
        self.model.put(("load_model",))
        self.request_image()

    def scroll_y(self, *args, **kwargs):
        """Scroll canvas vertically and redraw the image"""
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        """Scroll canvas horizontally and redraw the image"""
        print("SX ", args, kwargs)
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def move_from(self, event):
        """Remember previous coordinates for scrolling with the mouse"""
        self.prev_x = event.x
        self.prev_y = event.y

    def move_to(self, event):
        """Drag (move) canvas to the new position"""
        self.transform @= translation_matrix(
            (event.x - self.prev_x) / WIDTH, (event.y - self.prev_y) / HEIGHT
        )
        self.prev_x = event.x
        self.prev_y = event.y
        self.request_image()  # redraw the image

    def wheel(self, event):
        """Zoom with mouse wheel"""
        print(event)

        if event.num == 4 or event.delta == 120:  # scroll down
            self.transform @= translation_matrix(-event.x / WIDTH, -event.y / HEIGHT)
            self.transform @= scale_matrix(1 / 0.9, 1 / 0.9)
            self.transform @= translation_matrix(event.x / WIDTH, event.y / HEIGHT)
        if event.num == 5 or event.delta == -120:  # scroll up
            self.transform @= translation_matrix(-event.x / WIDTH, -event.y / HEIGHT)
            self.transform @= scale_matrix(0.9, 0.9)
            self.transform @= translation_matrix(event.x / WIDTH, event.y / HEIGHT)

        # print(event)
        # x = self.canvas.canvasx(event.x)
        # y = self.canvas.canvasy(event.y)
        # bbox = self.canvas.bbox(self.container)  # get image area
        # if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        # else: return  # zoom only inside image area
        # scale = 1.0
        # # Respond to Linux (event.num) or Windows (event.delta) wheel event
        # if event.num == 5 or event.delta == -120:  # scroll down
        #     i = min(self.width, self.height)
        #     if int(i * self.imscale) < 30: return  # image is less than 30 pixels
        #     self.imscale /= self.delta
        #     scale        /= self.delta
        # if event.num == 4 or event.delta == 120:  # scroll up
        #     i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
        #     if i < self.imscale: return  # 1 pixel is bigger than the visible area
        #     self.imscale *= self.delta
        #     scale        *= self.delta
        # self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.request_image()

    def check_queue(self):
        while self.inbox.qsize():
            head = self.inbox.get()
            self.show_image(head)
        self.master.after(100, self.check_queue)

    def request_image(self):
        def cb(img):
            self.inbox.put(img)

        self.model.put(
            (
                "render",
                self.transform,
                self.tx.get(),
                self.ty.get(),
                self.a.get(),
                self.b.get(),
                self.movement.get(),
                # self.circle1start.get(),
                # self.circle1end.get(),
                # self.cp.get(),
                cb,
            )
        )

    def show_image(self, img):
        """Show image on the Canvas"""
        self.img = Image.fromarray(img.numpy(), mode="RGBA")
        self.imagetk = ImageTk.PhotoImage(self.img)
        self.canvas["image"] = self.imagetk
        # imageid = self.canvas.create_image(
        #                                 anchor='nw', image=imagetk)
        # self.canvas.lower(imageid)  # set image into background
        # self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

        # bbox1 = self.canvas.bbox(self.container)  # get image area
        # # Remove 1 pixel shift at the sides of the bbox1
        # bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        # bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
        #          self.canvas.canvasy(0),
        #          self.canvas.canvasx(self.canvas.winfo_width()),
        #          self.canvas.canvasy(self.canvas.winfo_height()))
        # bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
        #         max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        # if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
        #     bbox[0] = bbox1[0]
        #     bbox[2] = bbox1[2]
        # if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
        #     bbox[1] = bbox1[1]
        #     bbox[3] = bbox1[3]
        # self.canvas.configure(scrollregion=bbox)  # set scroll region
        # x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        # y1 = max(bbox2[1] - bbox1[1], 0)
        # x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        # y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        # if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
        #     x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
        #     y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
        #     image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
        #     imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
        #     imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
        #                                        anchor='nw', image=imagetk)
        #     self.canvas.lower(imageid)  # set image into background
        #     self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection


q = queue.Queue()
q.put(("load_model",))

t = threading.Thread(target=lambda: worker(Collapser(q)), daemon=True)
t.start()


root = tk.Tk()
app = Zoom_Advanced(root, q)
root.mainloop()
