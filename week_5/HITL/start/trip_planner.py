"""
Trip Planning System with Human-in-the-Loop

This script demonstrates a simple HITL system for planning a trip with three steps:
1. Destination selection with LLM suggestion
2. Flight selection with LLM suggestion
3. Itinerary generation with LLM suggestion

Each step includes human verification and modification options.
"""

import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI
import os
from pydantic import BaseModel, Field

@dataclass
class TripPlan:
    """Represents a complete trip plan."""
    destination: str
    departure_city: str
    flight_info: Dict
    itinerary: Dict

class FlightInfo(BaseModel):
    """Model for flight information."""
    flight_number: str = Field(description="Flight number")
    departure_time: str = Field(description="Departure time in HH:MM format")
    arrival_time: str = Field(description="Arrival time in HH:MM format")
    airline: str = Field(description="Name of the airline")

class Itinerary(BaseModel):
    """Model for daily itinerary."""
    daily_activities: List[List[str]] = Field(description="List of activities for each day of the trip")

class TripPlanner:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.trip_plan = None

    def suggest_destination(self) -> str:
        """Use LLM to suggest a travel destination."""
        prompt = """Suggest an interesting travel destination. 
        Consider factors like weather, tourist attractions, and cultural experiences.
        Return only the destination name, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting destination suggestion: {e}")
            return "Paris"  # Fallback destination

    def suggest_flight(self, departure_city: str, destination: str) -> Dict:
        """Use LLM to suggest a fictional flight."""
        prompt = f"""Generate a fictional flight from {departure_city} to {destination}.
        Include departure time, arrival time, and flight number."""

        try:
            response = self.client.responses.parse(
                model="gpt-4o-mini-2024-07-18",
                input=[
                    {"role": "system", "content": "You are a helpful travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                text_format=FlightInfo
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error getting flight suggestion: {e}")
            return {
                "flight_number": "AA123",
                "departure_time": "10:00",
                "arrival_time": "12:00",
                "airline": "Example Airlines"
            }

    def generate_itinerary(self, destination: str) -> Dict:
        """Use LLM to generate a brief itinerary."""
        prompt = f"""Generate a brief 3-day itinerary for {destination}.
        Include 2-3 activities per day."""

        try:
            response = self.client.responses.parse(
                model="gpt-4o-mini-2024-07-18",
                input=[
                    {"role": "system", "content": "You are a helpful travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                text_format=Itinerary
            )
            return response.output_parsed
        except Exception as e:
            print(f"Error generating itinerary: {e}")
            return {
                "day1": ["Morning activity", "Afternoon activity"],
                "day2": ["Morning activity", "Afternoon activity"],
                "day3": ["Morning activity", "Afternoon activity"]
            }

    def get_human_confirmation(self, prompt: str) -> bool:
        """Get human confirmation for a suggestion."""
        while True:
            choice = input(f"{prompt} (y/n): ").lower().strip()
            if choice in ['y', 'n']:
                return choice == 'y'
            print("Please enter 'y' or 'n'")

    def plan_trip(self) -> None:
        """Run the complete trip planning process."""
        print("\n=== Welcome to the Trip Planner! ===\n")

        ## TODO Decide which steps would benefit from a human-in-the-loop experience.
        # Use get_human_confirmation to stop the flow and ask the human a question.

        # Step 1: Destination Selection
        suggested_destination = self.suggest_destination()
        print(f"\nSuggested destination: {suggested_destination}")
        
        # Example
        if self.get_human_confirmation("Would you like to use this destination?"):
            destination = suggested_destination
        else:
            destination = input("Enter your preferred destination: ").strip()
        
        # Step 2: Flight Selection
        departure_city = input("\nEnter your departure city: ").strip()
        
        # Get suggested flight
        suggested_flight = self.suggest_flight(departure_city, destination)
        
        # Display suggested flight
        print("\n=== Suggested Flight ===")
        print(f"Airline: {suggested_flight.airline}")
        print(f"Flight Number: {suggested_flight.flight_number}")
        print(f"Departure: {suggested_flight.departure_time}")
        print(f"Arrival: {suggested_flight.arrival_time}")
        
        # Ask for confirmation
        if self.get_human_confirmation("Would you like to use this flight?"):
            flight = suggested_flight
        else:
            print("\nPlease enter your flight details:")
            flight = FlightInfo(
                flight_number=input("Flight number (e.g., AA123): ").strip(),
                departure_time=input("Departure time (HH:MM): ").strip(),
                arrival_time=input("Arrival time (HH:MM): ").strip(),
                airline=input("Airline: ").strip()
            )
        
        # Step 3: Itinerary Generation
        suggested_itinerary = self.generate_itinerary(destination)
        
        def display_itinerary(itinerary):
            print("\n=== Your Itinerary ===")
            for day_num, activities in enumerate(itinerary.daily_activities, 1):
                print(f"\nDAY {day_num}:")
                for i, activity in enumerate(activities, 1):
                    print(f"  {i}. {activity}")
        
        while True:
            display_itinerary(suggested_itinerary)
            
            if self.get_human_confirmation("Would you like to keep this itinerary?"):
                break
                
            print("\nWhat would you like to do?")
            print("1. Add an activity")
            print("2. Remove an activity")
            print("3. Edit an activity")
            print("4. Keep as is")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                day = int(input("Enter day number: ")) - 1
                if 0 <= day < len(suggested_itinerary.daily_activities):
                    activity = input("Enter the activity to add: ").strip()
                    suggested_itinerary.daily_activities[day].append(activity)
                    print("Activity added!")
                else:
                    print("Invalid day number.")
                    
            elif choice == '2':
                day = int(input("Enter day number: ")) - 1
                if 0 <= day < len(suggested_itinerary.daily_activities):
                    activity_num = int(input("Enter activity number to remove: ")) - 1
                    if 0 <= activity_num < len(suggested_itinerary.daily_activities[day]):
                        removed = suggested_itinerary.daily_activities[day].pop(activity_num)
                        print(f"Removed: {removed}")
                    else:
                        print("Invalid activity number.")
                else:
                    print("Invalid day number.")
                    
            elif choice == '3':
                day = int(input("Enter day number: ")) - 1
                if 0 <= day < len(suggested_itinerary.daily_activities):
                    activity_num = int(input("Enter activity number to edit: ")) - 1
                    if 0 <= activity_num < len(suggested_itinerary.daily_activities[day]):
                        new_activity = input("Enter new activity: ").strip()
                        old_activity = suggested_itinerary.daily_activities[day][activity_num]
                        suggested_itinerary.daily_activities[day][activity_num] = new_activity
                        print(f"Changed '{old_activity}' to '{new_activity}'")
                    else:
                        print("Invalid activity number.")
                else:
                    print("Invalid day number.")
                    
            elif choice == '4':
                break
                
            else:
                print("Invalid choice. Please try again.")
        
        # Save the complete trip plan
        self.trip_plan = TripPlan(
            destination=destination,
            departure_city=departure_city,
            flight_info=flight,
            itinerary=suggested_itinerary
        )

        # Final confirmation
        print("\n=== Review Your Trip Plan ===")
        print(f"\nDestination: {self.trip_plan.destination}")
        print(f"Departure from: {self.trip_plan.departure_city}")
        print("\nFlight Information:")
        print(f"  Airline: {flight.airline}")
        print(f"  Flight: {flight.flight_number}")
        print(f"  Departure: {flight.departure_time}")
        print(f"  Arrival: {flight.arrival_time}")
        
        print("\nItinerary:")
        for day_num, activities in enumerate(self.trip_plan.itinerary.daily_activities, 1):
            print(f"\n  DAY {day_num}:")
            for i, activity in enumerate(activities, 1):
                print(f"    {i}. {activity}")
        
        if self.get_human_confirmation("\nWould you like to confirm and book this trip?"):
            print("\n✅ Your trip has been booked! Have a great journey! 🚀")
        else:
            print("\n❌ Trip planning cancelled. Your trip has not been booked.")
            return

def main():
    planner = TripPlanner()
    planner.plan_trip()

if __name__ == "__main__":
    main() 