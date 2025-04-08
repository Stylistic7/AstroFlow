import { Graphics } from '@pixi/react';

export default function Polygon({ points, color = 0xff0000, alpha = 0.5, lineWidth = 2, onClick }) {
    return (
        <Graphics
            draw={(g) => {
                g.clear();
                g.lineStyle(lineWidth, color, alpha);
                g.beginFill(color, alpha * 0.2);
                g.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < points.length; i++) {
                    g.lineTo(points[i][0], points[i][1]);
                }
                g.closePath();
                g.endFill();
            }}
            interactive={!!onClick}
            pointertap={onClick}
        />
    );
}
