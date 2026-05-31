import { useEffect, useRef } from "react";
import { DIRECTION_KEY_CODES, INTERACT_KEY_CODES, DEFAULT_ACTION_IDX } from "../constants.js";
import { removeValue } from "../utils.js";

export default function useBufferedInput(enabled) {
    const heldDirectionsRef = useRef([]);
    const interactBufferRef = useRef(0);

    useEffect(() => {
        if (!enabled) {
            return undefined;
        }
        function clearInputs() {
            heldDirectionsRef.current = [];
            interactBufferRef.current = 0;
        }
        function onKeyDown(event) {
            const directionAction = DIRECTION_KEY_CODES[event.which];
            if (directionAction !== undefined) {
                heldDirectionsRef.current = removeValue(heldDirectionsRef.current, directionAction).concat([directionAction]);
                event.preventDefault();
                return;
            }
            if (INTERACT_KEY_CODES[event.which]) {
                interactBufferRef.current = Math.min(interactBufferRef.current + 2, 4);
                event.preventDefault();
            }
        }
        function onKeyUp(event) {
            const directionAction = DIRECTION_KEY_CODES[event.which];
            if (directionAction === undefined) {
                return;
            }
            heldDirectionsRef.current = removeValue(heldDirectionsRef.current, directionAction);
        }
        document.addEventListener("keydown", onKeyDown);
        document.addEventListener("keyup", onKeyUp);
        window.addEventListener("blur", clearInputs);
        return function cleanup() {
            document.removeEventListener("keydown", onKeyDown);
            document.removeEventListener("keyup", onKeyUp);
            window.removeEventListener("blur", clearInputs);
        };
    }, [enabled]);

    function consumeAction() {
        if (interactBufferRef.current > 0) {
            interactBufferRef.current -= 1;
            return 5;
        }
        if (heldDirectionsRef.current.length) {
            return heldDirectionsRef.current[heldDirectionsRef.current.length - 1];
        }
        return DEFAULT_ACTION_IDX;
    }

    return { consumeAction };
}
