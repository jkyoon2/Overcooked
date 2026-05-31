import React from "react";
import { OBJECTS_ATLAS, OBJECTS_IMAGE_URL } from "../spa/atlas.js";
import ChefVisual from "./ChefVisual.jsx";
import IngredientStrip from "./IngredientStrip.jsx";
import AtlasSprite from "./AtlasSprite.jsx";

export default function CollaborateSceneAsset() {
    return (
        <div className="mode-scene mode-scene--collaborate">
            <div className="mode-scene__pair">
                <ChefVisual hatColor="gray" size={58} />
                <div className="mode-scene__plus">+</div>
                <ChefVisual hatColor="orange" size={58} />
            </div>
            <div className="mode-scene__pair">
                <IngredientStrip ingredients={["tomato", "tomato", "onion"]} />
                <AtlasSprite atlas={OBJECTS_ATLAS} imageUrl={OBJECTS_IMAGE_URL} frameName="soup-tomato-dish.png" size={46} className="mini-sprite" />
            </div>
            <div className="mode-scene__caption">Cook side by side with the AI chef.</div>
        </div>
    );
}
