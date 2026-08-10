use std::io ;
use rand::Rng; 
use std::cmp::Ordering;
fn main(){
    println!("Guess a number : ");
    let answer = rand::thread_rng().gen_range(1..=100);
    loop{
        println!("input your guess : ");
        let mut guess = String::new(); 
        io::stdin().read_line(&mut guess).expect("Failed to read line");
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Please enter a valid number");
                continue;
            }
        };
        match guess.cmp(&answer) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}