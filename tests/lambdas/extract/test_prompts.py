from lambdas.extract.prompts import build_extraction_prompt


def test_prompt_includes_title_and_description():
    prompt = build_extraction_prompt("Backend Engineer", "We need a Python developer.")

    assert "Backend Engineer" in prompt
    assert "We need a Python developer." in prompt


def test_prompt_lists_every_required_output_field():
    prompt = build_extraction_prompt("Engineer", "desc")

    for field in (
        "title",
        "company",
        "comp_min",
        "comp_max",
        "seniority",
        "required_skills",
        "remote_status",
        "visa_status",
    ):
        assert f'"{field}"' in prompt


def test_prompt_constrains_status_fields_to_the_schemas_literal_values():
    prompt = build_extraction_prompt("Engineer", "desc")

    assert '"remote", "hybrid", "onsite", "unknown"' in prompt
    assert '"sponsors", "no_sponsorship", "unknown"' in prompt
