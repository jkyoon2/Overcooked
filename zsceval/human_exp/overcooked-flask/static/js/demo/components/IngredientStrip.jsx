import React from "react";
import { OBJECTS_ATLAS, OBJECTS_IMAGE_URL } from "../spa/atlas.js";
import AtlasSprite from "./AtlasSprite.jsx";

export default function IngredientStrip(props) {
    return (
        <div className="ingredient-strip">
            {props.ingredients.map((ingredient, index) => (
                <React.Fragment key={ingredient + "-" + index}>
                    <AtlasSprite
                        atlas={OBJECTS_ATLAS}
                        imageUrl={OBJECTS_IMAGE_URL}
                        frameName={ingredient === "onion" ? "onion.png" : "tomato.png"}
                        size={38}
                        className="mini-sprite"
                    />
                    {index < props.ingredients.length - 1 ? <span className="ingredient-strip__plus">+</span> : null}
                </React.Fragment>
            ))}
        </div>
    );
}
