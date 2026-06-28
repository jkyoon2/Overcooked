import React, { useState, useEffect, useRef } from "react";
import OvercookedBoard from "../spa/board.jsx";
import ExpectationComposer from "../spa/expectation.jsx";
import { TIMESTEP_MS } from "../constants.js";
import { apiPost, makeTrajId } from "../utils.js";
import useBufferedInput from "../hooks/useBufferedInput.js";
import ModalShell from "./ModalShell.jsx";
import TrialStats from "./TrialStats.jsx";
import PostTrialRatingModal from "./PostTrialRatingModal.jsx";

export default function InteractiveTrialRunner(props) {
    const [runtime, setRuntime] = useState(null);
    const [status, setStatus] = useState("booting");
    const [error, setError] = useState("");
    const [probe, setProbe] = useState(null);
    const [probeSubmitted, setProbeSubmitted] = useState(false);
    const [probeDraftKey, setProbeDraftKey] = useState(0);
    const [finishPayload, setFinishPayload] = useState(null);
    const statusRef = useRef("booting");
    const input = useBufferedInput(props.trial.human_player_index !== null && props.trial.human_player_index !== undefined);

    useEffect(() => {
        statusRef.current = status;
    }, [status]);

    useEffect(() => {
        let mounted = true;
        apiPost("/start_trial", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
        }).then((startPayload) => {
            if (!mounted) { return; }
            setRuntime(startPayload.runtime);
            setStatus("running");
        }).catch((startError) => {
            if (!mounted) { return; }
            setError(String(startError.message || startError));
            setStatus("error");
        });
        return function cleanup() { mounted = false; };
    }, [props.trial.id, props.userInfo]);

    useEffect(() => {
        if (status !== "running") { return undefined; }
        let cancelled = false;
        let timerId = null;

        function schedule(delay) {
            timerId = window.setTimeout(stepLoop, delay);
        }

        function stepLoop() {
            if (cancelled || statusRef.current !== "running") { return; }
            const startedAt = window.performance.now();
            apiPost("/step_trial", {
                user_info: props.userInfo,
                trial_id: props.trial.id,
                human_action_idx: input.consumeAction(),
            }).then((response) => {
                if (cancelled) { return; }
                if (response.probe_pending) {
                    setRuntime(response.runtime);
                    setProbe(response.probe);
                    setProbeSubmitted(false);
                    setProbeDraftKey((current) => current + 1);
                    setStatus("probe");
                    return;
                }
                setRuntime(response.runtime);
                if (response.done) {
                    finishTrial();
                    return;
                }
                schedule(Math.max(0, TIMESTEP_MS - (window.performance.now() - startedAt)));
            }).catch((stepError) => {
                if (cancelled) { return; }
                setError(String(stepError.message || stepError));
                setStatus("error");
            });
        }

        schedule(0);
        return function cleanup() {
            cancelled = true;
            window.clearTimeout(timerId);
        };
    }, [status, props.trial.id, props.userInfo, input]);

    function finishTrial() {
        setStatus("finishing");
        apiPost("/finish_episode", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            traj_id: makeTrajId(props.trial.id),
            summary: { client: "react-spa" },
        }).then((payload) => {
            if (!props.trial.post_trial_questions || props.trial.post_trial_questions.length === 0) {
                props.onComplete(payload.trial_summary, payload.trajectory);
                return;
            }
            setFinishPayload(payload);
            setStatus("rating");
        }).catch((finishError) => {
            setError(String(finishError.message || finishError));
            setStatus("error");
        });
    }

    function submitProbe(expectation) {
        apiPost("/submit_probe", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            selected_subtask_id: expectation.selectedSubtaskId,
            extra: {
                expected_path: expectation.expectedPath,
                confidence: expectation.confidence,
                start_position: expectation.startPosition,
                input_mode: "step_wizard",
            },
        }).then((response) => {
            return apiPost("/save_trial_data", {
                user_info: props.userInfo,
                trial_id: props.trial.id,
                updates: { latest_probe: response.probe_record },
            });
        }).then(() => {
            setProbeSubmitted(true);
        }).catch((submitError) => {
            setError(String(submitError.message || submitError));
            setStatus("error");
        });
    }

    function resumeAfterProbe() {
        apiPost("/resume_trial_after_probe", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
        }).then((response) => {
            setRuntime(response.runtime);
            setProbe(null);
            setProbeSubmitted(false);
            if (response.done) { finishTrial(); return; }
            setStatus("running");
        }).catch((resumeError) => {
            setError(String(resumeError.message || resumeError));
            setStatus("error");
        });
    }

    function submitTrialRating(values) {
        const mergedSummary = Object.assign({}, finishPayload.trial_summary, { post_trial_rating: values });
        apiPost("/save_trial_data", {
            user_info: props.userInfo,
            trial_id: props.trial.id,
            updates: { post_trial_rating: values, trial_summary: mergedSummary },
        }).then(() => {
            props.onComplete(mergedSummary, finishPayload.trajectory);
        }).catch((ratingError) => {
            setError(String(ratingError.message || ratingError));
            setStatus("error");
        });
    }

    return (
        <div className="trial-screen">
            <section className="panel-card panel-card--board">
                <div className="trial-hero">
                    <div className="panel-eyebrow">{props.trial.title}</div>
                    <h3>{props.trial.mode === "observe" ? "Focus on the highlighted AI chef" : "Cook with the AI chef"}</h3>
                    <p>{props.trial.instruction}</p>
                    {props.trial.mode === "observe" ? <div className="callout">Watch the chef with the bold green focus box.</div> : null}
                </div>
                <TrialStats runtime={runtime || {}} />
                <OvercookedBoard
                    layoutGrid={props.trial.layout_grid}
                    state={runtime ? runtime.state : null}
                    trial={Object.assign({}, props.trial, { show_target_highlight: props.trial.mode === "observe" })}
                />
                {error ? <div className="callout callout--danger">{error}</div> : null}
            </section>

            <ModalShell open={status === "probe"} eyebrow="Probe" title={probeSubmitted ? "Expectation saved" : "What do you expect next?"}>
                {!probeSubmitted ? (
                    <ExpectationComposer
                        key={"probe-" + probeDraftKey}
                        title="Report what you expect from the AI chef"
                        probeIndex={probe ? probe.probe_index : 1}
                        probeTotal={probe ? probe.probe_total : props.trial.probe.count}
                        prompt={props.trial.probe.prompt}
                        sketchPrompt={props.trial.probe.sketch_prompt}
                        confidencePrompt={props.trial.probe.confidence_prompt}
                        subtaskOptions={props.subtaskOptions}
                        layoutGrid={props.trial.layout_grid}
                        state={runtime ? runtime.state : null}
                        targetAgentIndex={props.trial.target_agent_index}
                        onSubmit={submitProbe}
                    />
                ) : (
                    <div className="panel-stack">
                        <p>Now watch what the AI chef actually does.</p>
                        <div className="stage-actions">
                            <button className="primary-button" type="button" onClick={resumeAfterProbe}>Resume</button>
                        </div>
                    </div>
                )}
            </ModalShell>

            <PostTrialRatingModal
                open={status === "rating" && Boolean(finishPayload)}
                trial={props.trial}
                initialValues={{}}
                onSubmit={submitTrialRating}
            />
        </div>
    );
}
