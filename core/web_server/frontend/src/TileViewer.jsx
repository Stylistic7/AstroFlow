import { useEffect, useRef, useState } from 'react';
import * as PIXI from 'pixi.js';
import { Stage, useApp } from '@pixi/react';
import { useParams } from 'react-router-dom';
import { Viewport } from 'pixi-viewport';

const TILE_SIZE = 256;

const TileViewer = () => {
  const { imageHash } = useParams();
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    fetch(`/tile_metadata/${imageHash}`)
      .then((res) => res.json())
      .then((meta) => setMetadata(meta));
  }, [imageHash]);

  if (!metadata) return null;

  const worldWidth = metadata.tileWidth * Math.pow(2, metadata.maxZoom);
  const worldHeight = metadata.tileHeight * Math.pow(2, metadata.maxZoom);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Stage
        width={window.innerWidth}
        height={window.innerHeight}
        options={{ backgroundColor: 0x000000 }}
      >
        <ImperativeViewport
          imageHash={imageHash}
          imageWidth={metadata.width}
          imageHeight={metadata.height}
          worldWidth={worldWidth}
          worldHeight={worldHeight}
          maxZoom={metadata.maxZoom}
        />
      </Stage>
    </div>
  );
};

const ImperativeViewport = ({ imageHash, imageWidth, imageHeight, worldWidth, worldHeight, maxZoom }) => {
  const app = useApp();
  const viewportRef = useRef(null);
  const spritesRef = useRef({});

  useEffect(() => {
    const rafId = requestAnimationFrame(() => {
      const screenWidth = app.renderer.view.clientWidth;
      const screenHeight = app.renderer.view.clientHeight;

      const viewport = new Viewport({
        screenWidth,
        screenHeight,
        worldWidth,
        worldHeight,
        events: app.renderer.events,
        interaction: app.renderer.plugins.interaction,
        passiveWheel: false,
      });

      viewport.drag().pinch().decelerate();
      setTimeout(() => viewport.wheel(), 0);

      const fitScale = Math.min(
        screenWidth / imageWidth,
        screenHeight / imageHeight
      );

      viewport.clampZoom({ minScale: fitScale, maxScale: 1 });
      viewport.setZoom(fitScale, true);
      viewport.moveCenter(imageWidth / 2, imageHeight / 2);

      viewportRef.current = viewport;
      app.stage.addChild(viewport);

      const updateClamp = () => {
        const scale = viewport.scale.x;
        const halfWidth = (screenWidth / scale) / 2;
        const halfHeight = (screenHeight / scale) / 2;

        viewport.clamp({
          left: -halfWidth,
          right: imageWidth + halfWidth,
          top: -halfHeight,
          bottom: imageHeight + halfHeight,
        });
      };

      const updateTiles = () => {
        const scale = viewport.scale.x;
        const pixelsPerScreenPixel = 1 / scale;
        let bestZ = Math.floor(maxZoom - Math.log2(pixelsPerScreenPixel));
        bestZ = Math.max(0, Math.min(maxZoom, bestZ));

        const levelScale = Math.pow(2, maxZoom - bestZ);
        const scaledWidth = Math.ceil(imageWidth / levelScale);
        const scaledHeight = Math.ceil(imageHeight / levelScale);

        const tilesX = Math.ceil(scaledWidth / TILE_SIZE);
        const tilesY = Math.ceil(scaledHeight / TILE_SIZE);

        const bounds = viewport.getVisibleBounds();
        const xStart = Math.floor(bounds.x / (TILE_SIZE * levelScale));
        const yStart = Math.floor(bounds.y / (TILE_SIZE * levelScale));
        const xEnd = Math.ceil((bounds.x + bounds.width) / (TILE_SIZE * levelScale));
        const yEnd = Math.ceil((bounds.y + bounds.height) / (TILE_SIZE * levelScale));

        for (let x = xStart; x < xEnd; x++) {
          for (let y = yStart; y < yEnd; y++) {
            if (x < 0 || y < 0 || x >= tilesX || y >= tilesY) continue;
            const key = `${bestZ}/${x}/${y}`;
            if (spritesRef.current[key]) continue;

            const url = `/tiles/${imageHash}/${bestZ}/${x}/${y}.png`;
            PIXI.Texture.fromURL(url)
              .then((texture) => {
                const sprite = new PIXI.Sprite(texture);
                sprite.x = x * TILE_SIZE * levelScale;
                sprite.y = y * TILE_SIZE * levelScale;
                sprite.width = TILE_SIZE * levelScale;
                sprite.height = TILE_SIZE * levelScale;
                sprite.zIndex = bestZ;
                viewport.addChild(sprite);
                viewport.sortChildren();
                spritesRef.current[key] = sprite;
              })
              .catch((err) => {
                console.warn(`Failed to load tile: ${url}`, err);
              });
          }
        }

        updateClamp();
      };

      updateTiles();
      viewport.on('moved', updateTiles);
      viewport.on('zoomed', updateTiles);
      viewport.on('moved-end', updateTiles);
      viewport.on('zoomed-end', updateTiles);

      return () => {
        viewport.destroy();
        Object.values(spritesRef.current).forEach((sprite) => sprite.destroy());
        spritesRef.current = {};
      };
    });

    return () => cancelAnimationFrame(rafId);
  }, [app, imageHash, imageWidth, imageHeight, maxZoom, worldWidth, worldHeight]);

  return null;
};

export default TileViewer;
