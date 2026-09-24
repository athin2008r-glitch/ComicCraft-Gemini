from pydantic import BaseModel, Field, field_validator


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=4000)
    character_name: str = Field(min_length=1, max_length=80)
    setting: str = Field(min_length=1, max_length=40)
    tone: str = Field(min_length=1, max_length=40)
    art_style: str = Field(min_length=1, max_length=40)

    @field_validator("prompt", "character_name", "setting", "tone", "art_style")
    @classmethod
    def strip_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be empty.")
        return value
