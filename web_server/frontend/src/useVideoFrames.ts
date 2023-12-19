import React, { useEffect, useRef, useState } from "react";

type VideoEventListenerMap = {
  [EventName in keyof HTMLMediaElementEventMap]?: EventListener;
};

const useVideoFrames = (
  frameCallback = (videoTime: number) => {}
): [HTMLVideoElement | null, React.RefCallback<HTMLVideoElement>] => {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);

  const callbackRef = useRef(frameCallback);
  callbackRef.current = frameCallback;

  useEffect(() => {
    if (!video) return;

    let frameId: number | null;
    let requestFrame = requestAnimationFrame;
    let cancelFrame = cancelAnimationFrame;

    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      // https://web.dev/requestvideoframecallback-rvfc/
      const vid = video as HTMLVideoElement & {
        requestVideoFrameCallback: typeof requestAnimationFrame;
        cancelVideoFrameCallback: typeof cancelAnimationFrame;
      };
      console.log('video', video);
      console.log('video.requestVideoFrameCallback', video.requestVideoFrameCallback)
      console.log('vid.requestVideoFrameCallback', vid.requestVideoFrameCallback);
      requestFrame = vid.requestVideoFrameCallback.bind(vid);
      cancelFrame = vid.cancelVideoFrameCallback.bind(vid);
    } else {
      console.log('Video element does not support requestVideoFrameCallback');
    }

    const callbackFrame = (now: number, metadata?: any) => {
      const videoTime = metadata?.presentedFrames ?? video.currentTime;
      // const videoTime = video.getVideoPlaybackQuality().totalVideoFrames;
      callbackRef.current(videoTime);
      frameId = requestFrame(callbackFrame);
    };

    const eventListeners: VideoEventListenerMap = {
      loadeddata() {
        requestAnimationFrame(() => callbackRef.current(video.currentTime));
      },
      play() {
        frameId = requestFrame(callbackFrame);
      },
      pause() {
        cancelFrame(frameId ?? 0);
        frameId = null;
      },
      timeupdate() {
        if (!frameId) {
          requestAnimationFrame(() => callbackRef.current(video.currentTime));
        }
      },
    };

    Object.keys(eventListeners).forEach((eventName) => {
      const eventListener =
        eventListeners[eventName as keyof HTMLMediaElementEventMap];
      if (eventListener != null) {
        video.addEventListener(eventName, eventListener);
      }
    });

    return () => {
      cancelFrame(frameId ?? 0);

      Object.keys(eventListeners).forEach((eventName) => {
        const eventListener =
          eventListeners[eventName as keyof HTMLMediaElementEventMap];
        if (eventListener != null) {
          video.removeEventListener(eventName, eventListener);
        }
      });
    };
  }, [video]);

  return [video, setVideo];
};

export default useVideoFrames;
