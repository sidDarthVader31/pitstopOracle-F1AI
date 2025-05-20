
import {
  Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn,
  CreateDateColumn, UpdateDateColumn
} from "typeorm";
import { Season } from "./Season";

@Entity("race")
export class Race {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @ManyToOne(() => Season)
  @JoinColumn()
  season: Season;

  @Column()
  name: string;

  @Column()
  location: string;

  @Column({ type: "date" })
  date: Date;

  @Column("jsonb", { nullable: true })
  weather: any;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
