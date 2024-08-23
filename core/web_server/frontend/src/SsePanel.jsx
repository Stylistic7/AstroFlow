import React, { useEffect, useContext } from "react";
import DisambiguationPanel from "./DisambiguationPanel.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

export default function SsePanel() {
    const { stacks, targets, mosaics } = useContext(AstroFlowContext);
    const [data, setData] = React.useState();

    useEffect(() => {
        setData('foo');

        // TODO: This removes the hostname's port, which is a little gross.
        const stream_url = window.location.protocol + '//' + window.location.hostname + '/stream';
        const sse = new window.EventSource(stream_url);
        function getRealtimeData(d) {
            setData(d);
        }
        sse.onmessage = e => getRealtimeData(e.data);
        sse.addEventListener("foo", e => getRealtimeData(e.data));
        sse.onerror = () => {
            console.log('sse error');
            sse.close();
        }
        return () => {
            console.log('closing sse');
            sse.close();
        };
    }, []);


    return (
        <div>
            <span>SSE data: {data}</span>
        </div>
    );
}


