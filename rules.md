# Creating Rules

The rules for adjusting the CPU governor are defined in a JSON file. Each rule is an object in the `rules` array in the JSON file.

Here's the structure of a rule:

```json
{
    "usage_comparison": "<comparison_type>",
    "usage_lower_bound": <lower_bound>,
    "usage_upper_bound": <upper_bound>,
    "power_cost_comparison": "<comparison_type>",
    "power_cost_value": <value>,
    "governor": "<governor>"
}
```

### Rule Fields

- `usage_comparison`: This field specifies how the current CPU usage is compared to the bounds. It can be one of the following values:
    - `"higher_then"`: The usage is higher than the lower bound.
    - `"lower_then"`: The usage is lower than the lower bound.
    - `"between"`: The usage is between the lower and upper bounds.
    - `"default"`: This rule is used if no other rule matches.
- `usage_lower_bound` and `usage_upper_bound`: These fields specify the lower and upper bounds for the CPU usage. They are used when `usage_comparison` is `"higher_then"`, `"lower_then"`, or `"between"`.
- `power_cost_comparison`: This field specifies how the current power cost is compared to the value. It can be either `"higher_then"` or `"lower_then"`.
- `power_cost_value`: This field specifies the value to compare the current power cost to.
- `governor`: This field specifies the CPU governor to use if the rule matches.

### Example Rule

Here's an example of a rule:

\```json
{
    "usage_comparison": "higher_then",
    "usage_lower_bound": 50,
    "power_cost_comparison": "lower_then",
    "power_cost_value": 0.5,
    "governor": "performance"
}
\```

This rule states that if the CPU usage is higher than 50% and the power cost is lower than 0.5, then the CPU governor should be set to "performance".
