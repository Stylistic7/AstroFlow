# TODO: This filename should be saved in the Stack.
def get_best_light_frame(stack):
    if stack.master and os.path.exists(stack.master):
        print(f"Using stack master image for stack {stack}")
        return file_io.read_xisf_image(stack.master)

    print(f"Using single light frame thumbnail for stack {stack}")
    thumbnails = [cache[("thumbnail_image", f)] for f in stack.light_frames]
    index = np.argmax(map(np.std, thumbnails))
    return stack.light_frames[index]
