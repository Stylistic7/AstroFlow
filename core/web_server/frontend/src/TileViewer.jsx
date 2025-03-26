import { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Stage, useApp } from '@pixi/react';
import { useParams } from 'react-router-dom';
import { Viewport } from 'pixi-viewport';

if (!PIXI.EventSystem.prototype.__astroflow_patched__) {
  const original = PIXI.EventSystem.prototype.mapPositionToPoint;
  PIXI.EventSystem.prototype.mapPositionToPoint = function (point, x, y) {
    if (!this.domElement || !this.domElement.isConnected) return;
    return original.call(this, point, x, y);
  };
  PIXI.EventSystem.prototype.__astroflow_patched__ = true;
}


const TILE_SIZE = 256;

const TileViewer = ({ maxZoom = 3 }) => {
  const { imageHash } = useParams();
  const worldWidth = TILE_SIZE * Math.pow(2, maxZoom);
  const worldHeight = TILE_SIZE * Math.pow(2, maxZoom);

  return (
    <Stage
      width={800}
      height={600}
      options={{ backgroundColor: 0x222222 }}
    >
      <ImperativeViewport
        imageHash={imageHash}
        worldWidth={worldWidth}
        worldHeight={worldHeight}
        maxZoom={maxZoom}
      />
    </Stage>
  );
};

const ImperativeViewport = ({ imageHash, worldWidth, worldHeight, maxZoom }) => {
  const app = useApp();
  const viewportRef = useRef(null);
  const spritesRef = useRef({});

  useEffect(() => {
    if (!app.renderer.view || !app.renderer.events?.rootBoundary) return;

    const rafId = requestAnimationFrame(() => {
      const viewport = new Viewport({
        screenWidth: app.renderer.width,
        screenHeight: app.renderer.height,
        worldWidth,
        worldHeight,
        events: app.renderer.events,
        interaction: app.renderer.plugins.interaction,
        passiveWheel: false,
      });

      viewport.drag().pinch().decelerate();
      setTimeout(() => viewport.wheel(), 1000); // ✅ Delay wheel activation to avoid crash

      viewport.clampZoom({ minScale: 0.5, maxScale: 4 });
      viewport.moveCenter(worldWidth / 2, worldHeight / 2);

      viewportRef.current = viewport;
      app.stage.addChild(viewport);

      const updateTiles = () => {
        const z = maxZoom;
        const bounds = viewport.getVisibleBounds();
        const xStart = Math.floor(bounds.x / TILE_SIZE);
        const yStart = Math.floor(bounds.y / TILE_SIZE);
        const xEnd = Math.ceil((bounds.x + bounds.width) / TILE_SIZE);
        const yEnd = Math.ceil((bounds.y + bounds.height) / TILE_SIZE);

        const newKeys = new Set();

        for (let x = xStart; x < xEnd; x++) {
          for (let y = yStart; y < yEnd; y++) {
            const key = `${z}/${x}/${y}`;
            newKeys.add(key);

            if (!spritesRef.current[key]) {
              const url = `/tiles/${imageHash}/${z}/${x}/${y}.png`;
              PIXI.Texture.fromURL(url).then((texture) => {
                const sprite = new PIXI.Sprite(texture);
                sprite.x = x * TILE_SIZE;
                sprite.y = y * TILE_SIZE;
                sprite.width = TILE_SIZE;
                sprite.height = TILE_SIZE;
                viewport.addChild(sprite);
                spritesRef.current[key] = sprite;
              }).catch((err) => {
                console.warn(`Failed to load tile: ${url}`, err);
              });
            }
          }
        }

        for (const key in spritesRef.current) {
          if (!newKeys.has(key)) {
            const sprite = spritesRef.current[key];
            viewport.removeChild(sprite);
            sprite.destroy();
            delete spritesRef.current[key];
          }
        }
      };

      updateTiles();
      viewport.on('moved-end', updateTiles);
      viewport.on('zoomed-end', updateTiles);

      // Cleanup
      return () => {
        viewport.destroy();
        Object.values(spritesRef.current).forEach((sprite) => sprite.destroy());
        spritesRef.current = {};
      };
    });

    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [app, imageHash, maxZoom, worldWidth, worldHeight]);

  return null;
};

export default TileViewer;