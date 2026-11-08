from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Geography: str = Field(..., min_length=1)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    IsActiveMember: int = Field(..., ge=0, le=1)

    Gender: str | None = None
    CreditScore: int | None = Field(None, ge=300, le=850)
    Tenure: int | None = Field(None, ge=0, le=10)
    HasCrCard: int | None = Field(None, ge=0, le=1)
    EstimatedSalary: float | None = Field(None, ge=0)

class PredictionOutput(BaseModel):
    prediction: int
    churn_probability: float
