/**
 * The "hidden HLG video" hack to coax Chrome's compositor into
 * extended-range (EDR) display mode for the whole page.
 *
 * Background: as of Chrome 149 / mid-2026, the WebGL spec's
 * `drawingBufferColorSpace` IDL enum still only accepts "srgb" and
 * "display-p3" — setting "rec2100-hlg" throws. So we have no direct
 * API to flag a WebGL canvas as HDR. The known workaround is a tiny
 * HEVC HLG video, present and playing on the page: macOS Quartz +
 * Chrome's compositor detect the HDR-encoded video, switch the page's
 * compositing surface into EDR, and (per Chrome's layer-composition
 * behavior at the time of writing) any pixel value >1.0 written by a
 * sibling canvas gets uplifted to extended luminance rather than
 * clipped at SDR.
 *
 * The base64 data URI below IS an HEVC stream with BT.2020 / ARIB-
 * STD-B67 (HLG) transfer metadata — that's the only thing about it
 * that matters; the visible frames are a 16×16 black square.
 *
 * Rendered as 1×1 at the corner so it's effectively invisible but
 * still a live playing video the compositor must honor. The video
 * MUST NOT be display:none or visibility:hidden — that would drop
 * its layer entirely and lose the EDR upgrade.
 */
const HDR_HLG_DATA_URI = "data:video/mp4;base64,AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAAAhmcmVlAAAAvG1kYXQAAAAfTgEFGkdWStxcTEM/lO/FETzRQ6gD7gAA7gIAA3EYgAAAAEgoAa8iNjAkszOL+e58c//cEe//0TT//scp1n/381P/RWP/zOW4QtxorfVogeh8nQDbQAAAAwAQMCcWUTAAAAMAAAMAAAMA84AAAAAVAgHQAyu+KT35E7gAADFgAAADABLQAAAAEgIB4AiS76MTkNbgAAF3AAAPSAAAABICAeAEn8+hBOTXYAADUgAAHRAAAAPibW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAKcAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAw10cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAKcAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAACnAAAAAAABAAAAAAKFbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABdwAAAD6BVxAAAAAAAMWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABDb3JlIE1lZGlhIFZpZGVvAAAAAixtaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAHsc3RibAAAARxzdHNkAAAAAAAAAAEAAAEMaHZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAQABAASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAAHVodmNDAQIgAAAAsAAAAAAAPPAA/P36+gAACwOgAAEAGEABDAH//wIgAAADALAAAAMAAAMAPBXAkKEAAQAmQgEBAiAAAAMAsAAAAwAAAwA8oBQgQcCTDLYgV7kWVYC1CRAJAICiAAEACUQBwChkuNBTJAAAAApmaWVsAQAAAAATY29scm5jbHgACQAQAAkAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAAAAAALPwAACz8AAAAKHN0dHMAAAAAAAAAAwAAAAIAAAPoAAAAAQAAAAEAAAABAAAD6AAAABRzdHNzAAAAAAAAAAEAAAABAAAAEHNkdHAAAAAAIBAQGAAAAChjdHRzAAAAAAAAAAMAAAABAAAAAAAAAAEAAAfQAAAAAgAAAAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAQAAAABAAAAJHN0c3oAAAAAAAAAAAAAAAQAAABvAAAAGQAAABYAAAAWAAAAFHN0Y28AAAAAAAAAAQAAACwAAABhdWR0YQAAAFltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAACxpbHN0AAAAJKl0b28AAAAcZGF0YQAAAAEAAAAATGF2ZjYwLjMuMTAw";


/**
 * Renders as a full-size HDR backdrop behind whatever sibling element
 * the caller composites on top with `mix-blend-mode`. The video's HDR
 * luminance shines through bright pixels of the overlay; dark pixels
 * mask it. That makes the visible image's white parts emit at
 * extended-range luminance even though the overlay itself is a
 * plain-SDR canvas.
 */
export default function HdrHeadroomVideo() {
    return (
        <video
            src={HDR_HLG_DATA_URI}
            muted
            autoPlay
            loop
            playsInline
            controls={false}
            style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                pointerEvents: 'none',
            }}
        />
    );
}
