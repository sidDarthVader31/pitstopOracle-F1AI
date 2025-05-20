
import {
  Entity, PrimaryGeneratedColumn, ManyToOne, JoinColumn,
  CreateDateColumn, UpdateDateColumn
} from "typeorm";
import { Driver } from "./Driver";
import { Season } from "./Season";
import { Constructor } from "./Constructor";
import { Car } from "./Car";

@Entity("driver_season")
export class DriverSeason {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @ManyToOne(() => Driver)
  @JoinColumn()
  driver: Driver;

  @ManyToOne(() => Season)
  @JoinColumn()
  season: Season;

  @ManyToOne(() => Constructor)
  @JoinColumn()
  constructorTeam: Constructor;

  @ManyToOne(() => Car)
  @JoinColumn()
  car: Car;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
