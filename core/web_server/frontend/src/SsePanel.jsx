import { useEffect, useState } from "react";

export default function SsePanel() {
    const [data, setData] = useState();

    useEffect(() => {
        setData('foo');

        const stream_url = window.location.protocol + '//' + window.location.hostname + '/stream';
        const sse = new window.EventSource(stream_url);
        function getRealtimeData(d) {
            setData(d);
        }
        sse.onmessage = e => getRealtimeData(e.data);
        sse.addEventListener("foo", e => getRealtimeData(e.data));
        sse.onerror = () => {
            sse.close();
        }
        return () => {
            sse.close();
        };
    }, []);

    return (
        <div>
            <span>SSE data: {data}</span>
        </div>
    );
}
