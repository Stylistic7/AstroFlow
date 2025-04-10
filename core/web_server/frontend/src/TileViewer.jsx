import { useEffect, useRef, useState, useLayoutEffect } from 'react';
import * as PIXI from 'pixi.js';
import { Stage, useApp } from '@pixi/react';
import { useParams } from 'react-router-dom';
import { Viewport } from 'pixi-viewport';
import { Container } from '@pixi/react';
import Polygon from './Polygon.jsx';
import React, { useContext } from "react";

import AstroFlowContext from './AstroFlowContext.jsx';


const TILE_SIZE = 256;

const TileViewer = ({ objectType: objectTypeProp, imageHash: imageHashProp, filter: filterProp }) => {
  const { objectType: objectTypeParam, imageHash: imageHashParam, filter: filterParam } = useParams();

  const objectType = objectTypeProp ?? objectTypeParam;
  const imageHash = imageHashProp ?? imageHashParam;
  const filter = filterProp ?? filterParam;

  const [metadata, setMetadata] = useState(null);
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  const { mosaics } = useContext(AstroFlowContext);

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(() => {
      const bounds = containerRef.current.getBoundingClientRect();
      setContainerSize({ width: bounds.width, height: bounds.height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    fetch(`/tile_metadata/${objectType}/${imageHash}/${filter}`)
      .then((res) => res.json())
      .then((meta) => setMetadata(meta));
  }, [objectType, imageHash]);

  if (!metadata || !containerRef.current) return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;

  const worldWidth = metadata.tileWidth * Math.pow(2, metadata.maxZoom);
  const worldHeight = metadata.tileHeight * Math.pow(2, metadata.maxZoom);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
      <Stage
        width={containerSize.width}
        height={containerSize.height}
        options={{
          backgroundColor: 0x000000,
          resizeTo: containerRef.current,
        }}
        style={{ width: '100%', height: '100%' }}
      >
        <ImperativeViewport
          objectType={objectType}
          imageHash={imageHash}
          filter={filter}
          imageWidth={metadata.width}
          imageHeight={metadata.height}
          worldWidth={worldWidth}
          worldHeight={worldHeight}
          maxZoom={metadata.maxZoom}
          mosaics={mosaics}
        />
      </Stage>
    </div>
  );
};

// // Add this above your `ImperativeViewport` component return:
// const polygons = [
//   {
//     id: 'example-1',
//     points: [[1000, 1000], [2000, 1000], [1500, 2000]],
//     color: 0x00ff00,
//   },
//   // Add more as needed
// ];

const mosaicOutline = (mosaic) => {
  const projection = 'mercator';

  return [
    {
      id: 'mosaic.footprint',
      points: mosaic.projected_footprints[projection].coords,
      color: 0x00ff00,
    },
  ];
}

const ImperativeViewport = ({ objectType, imageHash, filter, imageWidth, imageHeight, worldWidth, worldHeight, maxZoom, mosaics }) => {
  const app = useApp();
  const viewportRef = useRef(null);
  const spritesRef = useRef({});

  useEffect(() => {
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
      const halfWidth = screenWidth / (2 * scale);
      const halfHeight = screenHeight / (2 * scale);

      viewport.clamp({
        left: -halfWidth,
        right: imageWidth + halfWidth,
        top: -halfHeight,
        bottom: imageHeight + halfHeight,
      });
    };

    const updatePolygons = () => {
      // console.log('mosaics ', mosaics);
      // if (objectType != "mosaic" || !mosaics) {
      //   return;
      // }

      let polygons = mosaicOutline(mosaics[imageHash]);
      console.log('displaying ', polygons.length, ' polygons');

      // Create polygon graphics and add them
      for (const poly of polygons) {
        console.log('polygon has ', poly.points.length, ' points');

        const g = new PIXI.Graphics();
        g.zIndex = 9999;
        g.lineStyle(2, poly.color ?? 0xff0000, 1);
        // g.beginFill(poly.color ?? 0xff0000, 0.2);
        g.moveTo(poly.points[0][0], poly.points[0][1]);
        for (let i = 1; i < poly.points.length; i++) {
          console.log('point[', i, '] ', poly.points[i]);
          g.lineTo(poly.points[i][0] * 5000, poly.points[i][1] * 5000);
        }
        g.closePath();
        // g.endFill();

        // Optional: make interactive
        g.interactive = true;
        g.cursor = 'pointer';
        g.on('pointertap', () => {
          console.log(`Clicked polygon ${poly.id}`);
        });

        viewport.addChild(g);
      }

      viewport.sortChildren();
    }

    const updateTiles = () => {
      const scale = viewport.scale.x;
      const pixelsPerScreenPixel = 1 / scale;
      let bestZ = Math.ceil(maxZoom - Math.log2(pixelsPerScreenPixel));
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

          const url = `/tile/${objectType}/${imageHash}/${filter}/${bestZ}/${x}/${y}.png`;
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

    const updateAll = () => {
      updateTiles();
      // updatePolygons();
    }

    updateAll();
    viewport.on('moved', updateAll);
    viewport.on('zoomed', updateAll);
    viewport.on('moved-end', updateAll);
    viewport.on('zoomed-end', updateAll);

    return () => {
      viewport.destroy();
      Object.values(spritesRef.current).forEach((sprite) => sprite.destroy());
      spritesRef.current = {};
    };
  }, [app, mosaics, imageHash, imageWidth, imageHeight, maxZoom, worldWidth, worldHeight]);

  return null;
};

export default TileViewer;
