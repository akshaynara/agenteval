import pytest

from agenteval import engine
from agenteval.suites.language import ASR_COMMON, asr_noise, profile_info
from agenteval.targets import SimulatedTarget


def test_asr_noise_is_deterministic_and_hits_numbers_and_negations():
    text = "I did not make the fifteen dollar payment"
    a = asr_noise(text, "indian_english", 1.0, seed="x")
    b = asr_noise(text, "indian_english", 1.0, seed="x")
    assert a == b  # same seed → same noise → comparable reruns
    assert "fifty" in a and "not" not in a.split()
    assert asr_noise(text, "indian_english", 0.0, seed="x") == text


def test_asr_noise_keeps_punctuation_attached_and_varies_by_seed():
    text = "Wait, I paid fifteen."
    outs = {asr_noise(text, "indian_english", 0.6, seed=str(i)) for i in range(20)}
    assert len(outs) > 1
    assert any("weight" in o.lower() for o in outs)


def test_unknown_profiles_still_work():
    info = profile_info("scottish_english")
    assert info["label"] == "Scottish English" and "Scottish English" in info["description"]
    assert ("fifteen", "fifty") in ASR_COMMON


def test_language_suite_finds_code_switching_failure(fake_llm):
    target = SimulatedTarget(fake_llm, "You are a bank agent.")
    events = []
    report = engine.run_language(
        fake_llm, target, "Banking", num_scenarios=2, turns=2,
        accents=["indian_english", "british_english"], code_switch=["hinglish"], asr=["indian_english"],
        on_event=lambda n, d: events.append(n),
    )
    # control + 2 accents + 1 code-switch + 1 ASR = 5 variants per scenario
    assert report.total_scenarios == 2 and report.total_variants == 10
    assert report.total_failures == 2  # the agent refuses Hinglish in both scenarios
    f = report.findings[0]
    assert f.failed and f.severity == "high" and f.bias_detected
    assert f.failing_variants == ["Hinglish"] and not f.control_failed
    assert report.average_consistency == 55
    sc = report.scenarios[0]
    kinds = [r.variant.kind for r in sc.results]
    assert kinds == ["control", "accent", "accent", "code_switch", "asr"]
    hinglish = next(r for r in sc.results if r.variant.profile == "hinglish")
    assert not hinglish.handled_correctly and "mera" in hinglish.transcript[0].content
    asr = next(r for r in sc.results if r.variant.kind == "asr")
    assert asr.variant.opening_message != sc.scenario.canonical_message  # noise applied
    assert set(report.profiles_tested) == {"indian_english", "british_english", "hinglish"}
    assert events.count("language_result") == 2 and events[-1] == "language_report"


def test_language_suite_passes_when_everyone_is_served(fake_llm):
    target = SimulatedTarget(fake_llm, "You are a bank agent.")
    report = engine.run_language(fake_llm, target, "Banking", 2, 2, accents=["british_english"])
    assert report.total_failures == 0 and report.average_consistency == 100


def test_language_needs_a_profile(fake_llm):
    with pytest.raises(ValueError, match="at least one"):
        engine.run_language(fake_llm, SimulatedTarget(fake_llm, "x"), "Banking", 1, 1, accents=[], code_switch=[], asr=[])


def test_language_findings_count_toward_the_gate(fake_llm):
    target = SimulatedTarget(fake_llm, "You are a bank agent.")
    report = engine.run(llm=fake_llm, target=target, name="t", domain="Banking", wind_tunnel_opts=None,
                        red_team_opts=None, language_opts={"scenarios": 2, "code_switch": ["hinglish"]},
                        fail_on="high")
    assert report.language and report.summary.severity_counts["high"] == 2
    assert not report.gate.passed


def test_asr_noise_always_changes_something_when_it_can():
    for i in range(50):
        assert asr_noise("I need to update my address", "american_english", 0.1, seed=str(i)) != \
            "I need to update my address"
    # nothing the error table knows: the text can only lose punctuation/case
    assert asr_noise("Hello there!", "american_english", 0.3, seed="x") in ("Hello there!", "hello there")
